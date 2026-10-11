"""Tests for Sprint 10.1 steps E (SPY benchmark/metrics) and F (operational API)."""

import json
import math
import os
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from datetime import date, timedelta
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from evaluation.benchmark import (
    BenchmarkError, compute_performance, load_benchmark, save_benchmark, validate_price_series,
)
from execution.paper_gateway import OrderRequest, OrderStatus, PaperExecutionGateway
from monitoring.api import ApiContext, make_handler, route
from portfolio.ledger import PortfolioLedger
from portfolio.risk_guard import RiskGuard

D0 = date(2026, 1, 5)


def series(vals, start=D0):
    return [(start + timedelta(days=i), v) for i, v in enumerate(vals)]


class TestBenchmarkMetrics(unittest.TestCase):
    def test_hand_computed_returns_and_drawdown(self):
        eq = series([100, 110, 99, 120])
        bm = series([100, 101, 102, 103])
        r = compute_performance(eq, bm)
        self.assertEqual(r["status"], "OK")
        self.assertAlmostEqual(r["cumulative_return"], 0.20, places=12)
        self.assertAlmostEqual(r["benchmark_return"], 0.03, places=12)
        self.assertAlmostEqual(r["cumulative_excess_return"], 0.17, places=12)
        self.assertAlmostEqual(r["max_drawdown"], 0.10, places=12)  # 110 -> 99

    def test_blocked_without_spy_and_no_substitution(self):
        r = compute_performance(series([100, 101, 102]), None)
        self.assertEqual(r["status"], "BLOCKED")
        self.assertNotIn("benchmark_return", r)
        self.assertIn("no substitute", r["reason"])

    def test_alignment_uses_intersection_only(self):
        eq = [(D0, 100.0), (D0 + timedelta(days=2), 110.0), (D0 + timedelta(days=3), 121.0)]
        bm = [(D0, 100.0), (D0 + timedelta(days=1), 500.0), (D0 + timedelta(days=3), 110.0)]
        r = compute_performance(eq, bm)
        self.assertEqual(r["coverage"]["aligned_dates"], 2)
        self.assertEqual(r["coverage"]["equity_dates_without_benchmark"], 1)
        self.assertEqual(r["coverage"]["benchmark_dates_outside_equity_window"], 1)
        self.assertAlmostEqual(r["benchmark_return"], 0.10, places=12)  # 500 outlier never used
        self.assertAlmostEqual(r["cumulative_return"], 0.21, places=12)

    def test_annualization_omitted_for_short_sample(self):
        r = compute_performance(series([100, 101, 102, 103]), series([100, 100, 101, 102]))
        self.assertIsNone(r["annualized_return"])
        self.assertTrue(any("annualized" in n for n in r["notes"]))

    def test_annualization_reported_for_long_sample(self):
        n = 253
        eq = series([100 * (1.001 ** i) for i in range(n)])
        bm = series([100 * (1.0005 ** i) for i in range(n)])
        r = compute_performance(eq, bm)
        self.assertAlmostEqual(r["annualized_return"], 1.001 ** 252 - 1, places=6)

    def test_turnover_costs_and_win_rate(self):
        tx = [{"shares": 10, "price": 100.0, "fee": 0.8}, {"shares": 10, "price": 110.0, "fee": 0.9}]
        r = compute_performance(series([1000, 1010, 1020]), series([100, 101, 102]),
                                transactions=tx, closed_pnls=[5.0, -2.0, 3.0])
        self.assertAlmostEqual(r["turnover_notional"], 2100.0)
        self.assertAlmostEqual(r["transaction_costs_total"], 1.7)
        self.assertAlmostEqual(r["win_rate"], 2 / 3)

    def test_constant_equity_has_no_sharpe(self):
        r = compute_performance(series([100, 100, 100]), series([100, 101, 102]))
        self.assertIsNone(r["sharpe_ratio"])
        self.assertTrue(math.isfinite(r["volatility_annualized"]))


class TestBenchmarkData(unittest.TestCase):
    def test_roundtrip_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            save_benchmark([(D0, 100.0), (D0 + timedelta(days=1), 101.5)], "unit-test", directory=d)
            rows, meta = load_benchmark(d)
            self.assertEqual(len(rows), 2)
            self.assertEqual(meta["source"], "unit-test")
            p = d / "SPY.csv"
            p.write_text(p.read_text().replace("101.5", "999.0"))
            with self.assertRaises(BenchmarkError):
                load_benchmark(d)

    def test_missing_file_raises(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(BenchmarkError):
                load_benchmark(Path(t))

    def test_invalid_series_rejected(self):
        for bad in ([], [(D0, 1.0), (D0, 2.0)], [(D0, 0.0)], [(D0, -5.0)], [(D0, float("nan"))]):
            with self.assertRaises(BenchmarkError):
                validate_price_series(bad)

    def test_fetch_retries_then_fails_without_fabricating(self):
        from evaluation import benchmark as bm
        fake = mock.Mock()
        fake.download.side_effect = TimeoutError("boom")
        with mock.patch.dict("sys.modules", {"yfinance": fake}), mock.patch.object(bm.time, "sleep"):
            with self.assertRaises(BenchmarkError):
                bm.fetch_spy("2026-01-01", "2026-02-01", retries=3)
        self.assertEqual(fake.download.call_count, 3)


class _StubMonitor:
    def __init__(self, ready=True):
        self.ready = ready

    def get_health(self):
        return {"status": "HEALTHY", "trading_mode": "PAPER_TRADING_ONLY"}

    def get_readiness(self):
        return {"ready": self.ready, "gating_issues": [] if self.ready else ["market data stale"]}

    def get_operational_metrics(self):
        return {"ok": True}


class _ApiBase(unittest.TestCase):
    TOKEN = "unit-test-token-123"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.guard = RiskGuard(state_file=Path(self.tmp.name) / "kill.json")
        self.orders = []
        self.ctx = ApiContext(
            risk_guard=self.guard, monitor=_StubMonitor(),
            orders_provider=lambda: self.orders,
            benchmark_report=Path(self.tmp.name) / "none.json",
        )
        p = mock.patch.dict(os.environ, {"TRADING_ADMIN_TOKEN": self.TOKEN})
        p.start()
        self.addCleanup(p.stop)

    def call(self, method, path, headers=None, body=None, query=None):
        return route(self.ctx, method, path, headers or {}, body, query)


class TestOpsApi(_ApiBase):
    def test_health_and_readiness(self):
        self.assertEqual(self.call("GET", "/health")[0], 200)
        self.assertEqual(self.call("GET", "/readiness")[0], 200)

    def test_readiness_503_when_not_ready(self):
        self.ctx.monitor = _StubMonitor(ready=False)
        status, body = self.call("GET", "/readiness")
        self.assertEqual(status, 503)
        self.assertFalse(body["ready"])

    def test_status_shows_paper_only_and_no_live(self):
        _, body = self.call("GET", "/trading/status")
        self.assertEqual(body["trading_mode"], "PAPER_TRADING_ONLY")
        self.assertFalse(body["live_trading_enabled"])
        self.assertTrue(body["accepting_new_orders"])

    def test_no_order_entry_endpoint_exists(self):
        for m, p in (("POST", "/orders"), ("PUT", "/orders"), ("POST", "/orders/submit")):
            self.assertEqual(self.call(m, p, {"X-Admin-Token": self.TOKEN}, {"ticker": "AAPL"})[0], 404)

    def test_kill_switch_requires_token(self):
        body = {"reason": "drill", "actor": "alice"}
        self.assertEqual(self.call("POST", "/trading/kill-switch/activate", {}, body)[0], 401)
        self.assertEqual(self.call("POST", "/trading/kill-switch/activate",
                                   {"X-Admin-Token": "wrong"}, body)[0], 401)
        self.assertFalse(self.guard.is_kill_switch_active()[0])

    def test_kill_switch_disabled_when_token_unset(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TRADING_ADMIN_TOKEN", None)
            status, _ = self.call("POST", "/trading/kill-switch/activate",
                                  {"X-Admin-Token": "x"}, {"reason": "r", "actor": "a"})
        self.assertEqual(status, 403)

    def test_kill_switch_requires_reason_and_actor(self):
        h = {"X-Admin-Token": self.TOKEN}
        self.assertEqual(self.call("POST", "/trading/kill-switch/activate", h, {"reason": "x"})[0], 400)
        self.assertEqual(self.call("POST", "/trading/kill-switch/activate", h, {"actor": "x"})[0], 400)

    def test_activate_blocks_orders_and_reactivate_is_audited(self):
        h = {"X-Admin-Token": self.TOKEN}
        gw = PaperExecutionGateway(ledger=PortfolioLedger(10000.0), risk_guard=self.guard)

        self.assertEqual(self.call("POST", "/trading/kill-switch/activate", h,
                                   {"reason": "market anomaly", "actor": "alice"})[0], 200)
        status_body = self.call("GET", "/trading/status")[1]
        self.assertTrue(status_body["kill_switch"]["active"])
        self.assertFalse(status_body["accepting_new_orders"])
        self.assertEqual(status_body["kill_switch"]["activated_by"], "API:alice")

        o = gw.submit_order(OrderRequest("AAPL", "BUY", 1.0, trade_date=D0), 100.0)
        self.assertEqual(o.status, OrderStatus.REJECTED)
        self.assertEqual(gw.ledger.cash, 10000.0)

        self.assertEqual(self.call("POST", "/trading/kill-switch/reactivate", h,
                                   {"reason": "cleared", "actor": "bob"})[0], 200)
        hist = self.call("GET", "/trading/kill-switch/history")[1]["history"]
        self.assertEqual([e["action"] for e in hist], ["ACTIVATE", "DEACTIVATE"])
        self.assertEqual([e["by"] for e in hist], ["API:alice", "API:bob"])

    def test_orders_filtering_and_lookup(self):
        self.orders = [
            {"order_id": "a1", "status": "FILLED", "trace_id": "t1", "ticker": "AAPL", "state_history": []},
            {"order_id": "b2", "status": "REJECTED", "trace_id": "t2", "ticker": "MSFT", "state_history": []},
        ]
        self.assertEqual(self.call("GET", "/orders")[1]["count"], 2)
        self.assertEqual(self.call("GET", "/orders", query={"status": "rejected"})[1]["count"], 1)
        self.assertEqual(self.call("GET", "/orders", query={"trace_id": "t1"})[1]["orders"][0]["order_id"], "a1")
        self.assertEqual(self.call("GET", "/orders/a1")[0], 200)
        self.assertEqual(self.call("GET", "/orders/zzz")[0], 404)

    def test_audit_merges_order_and_kill_switch_events_in_time_order(self):
        self.orders = [{"order_id": "a1", "trace_id": "t1", "ticker": "AAPL", "status": "FILLED",
                        "state_history": [{"status": "CREATED", "timestamp": "2000-01-01T00:00:00+00:00"}]}]
        self.guard.activate_kill_switch("r", "SYSTEM")
        events = self.call("GET", "/audit")[1]["events"]
        self.assertEqual([e["type"] for e in events], ["ORDER_TRANSITION", "KILL_SWITCH_ACTIVATE"])

    def test_benchmark_endpoint_blocked_when_absent(self):
        status, body = self.call("GET", "/benchmark")
        self.assertEqual(status, 404)
        self.assertEqual(body["status"], "BLOCKED")

    def test_token_never_in_any_response(self):
        h = {"X-Admin-Token": self.TOKEN}
        outs = [self.call("POST", "/trading/kill-switch/activate", h, {"reason": "r", "actor": "a"}),
                self.call("GET", "/trading/status"), self.call("GET", "/audit")]
        self.assertNotIn(self.TOKEN, json.dumps(outs, default=str))

    def test_unknown_route_404(self):
        self.assertEqual(self.call("GET", "/nope")[0], 404)


class TestOpsApiOverHttp(_ApiBase):
    def test_http_roundtrip_with_correlation_id(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.ctx))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        base = f"http://127.0.0.1:{server.server_address[1]}"

        req = urllib.request.Request(base + "/trading/status", headers={"X-Correlation-ID": "cid-42"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            body = json.loads(resp.read())
            self.assertEqual(resp.headers["X-Correlation-ID"], "cid-42")
        self.assertEqual(body["correlation_id"], "cid-42")
        self.assertEqual(body["trading_mode"], "PAPER_TRADING_ONLY")

        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(urllib.request.Request(
                base + "/trading/kill-switch/activate", data=b'{"reason":"r","actor":"a"}', method="POST"), timeout=5)
        self.assertEqual(cm.exception.code, 401)

        bad = urllib.request.Request(base + "/trading/kill-switch/activate", data=b"not json", method="POST",
                                     headers={"X-Admin-Token": self.TOKEN})
        with self.assertRaises(urllib.error.HTTPError) as cm2:
            urllib.request.urlopen(bad, timeout=5)
        self.assertEqual(cm2.exception.code, 400)


if __name__ == "__main__":
    unittest.main()
