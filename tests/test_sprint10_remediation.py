"""
Sprint 10.1 remediation tests: execution cost accounting, durability, atomicity,
concurrency and ledger reconciliation. Uses an isolated kill-switch state file and
a temp SQLite order store, so no shared state is touched.
"""

import tempfile
import threading
import unittest
from datetime import date
from pathlib import Path

from execution.order_store import OrderStore
from execution.paper_gateway import OrderRequest, OrderStatus, PaperExecutionGateway
from portfolio.ledger import PortfolioLedger
from portfolio.risk_guard import RiskGuard


class _Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.state_file = Path(self.tmp.name) / "kill.json"
        self.db_path = str(Path(self.tmp.name) / "orders.sqlite")

    def make_gateway(self, capital=10000.0, with_store=False, ledger=None):
        ledger = ledger or PortfolioLedger(initial_capital_usd=capital)
        guard = RiskGuard(state_file=self.state_file)
        store = OrderStore(self.db_path) if with_store else None
        if store:
            self.addCleanup(store.close)
        return PaperExecutionGateway(ledger=ledger, risk_guard=guard, store=store)


class TestCostAccounting(_Base):
    def test_buy_costs_applied_exactly_once(self):
        gw = self.make_gateway()
        o = gw.submit_order(OrderRequest("AAPL", "BUY", 10.0, trade_date=date(2026, 1, 5)), 100.0)
        self.assertEqual(o.status, OrderStatus.FILLED)
        fill = 100.0 * 1.0002
        notional = 10.0 * fill
        fees = notional * 0.0005 + notional * 0.0003  # commission + spread (slippage is in price)
        self.assertAlmostEqual(o.fill_price, fill, places=9)
        self.assertAlmostEqual(o.total_fees, fees, places=9)
        self.assertAlmostEqual(gw.ledger.cash, 10000.0 - notional - fees, places=9)

    def test_round_trip_cost_matches_hand_calculation(self):
        gw = self.make_gateway()
        gw.submit_order(OrderRequest("AAPL", "BUY", 10.0, trade_date=date(2026, 1, 5)), 100.0)
        gw.submit_order(OrderRequest("AAPL", "SELL", 10.0, trade_date=date(2026, 1, 6)), 100.0)
        buy_fill, sell_fill = 100.0 * 1.0002, 100.0 * 0.9998
        buy_n, sell_n = 10 * buy_fill, 10 * sell_fill
        expected = 10000.0 - buy_n - buy_n * 0.0008 + sell_n - sell_n * 0.0008
        self.assertAlmostEqual(gw.ledger.cash, expected, places=9)
        self.assertTrue(gw.ledger.reconcile()[0])


class TestLedgerAtomicity(_Base):
    def test_failed_buy_leaves_state_untouched(self):
        gw = self.make_gateway(capital=1000.0)
        before = (gw.ledger.cash, len(gw.ledger.transaction_history))
        with self.assertRaises(ValueError):
            gw.ledger.record_buy("AAPL", 100.0, 100.0, date(2026, 1, 5), {})
        self.assertEqual(before, (gw.ledger.cash, len(gw.ledger.transaction_history)))

    def test_duplicate_open_position_rejected_without_state_change(self):
        gw = self.make_gateway()
        gw.submit_order(OrderRequest("AAPL", "BUY", 5.0, trade_date=date(2026, 1, 5), idempotency_key="a"), 100.0)
        cash = gw.ledger.cash
        o = gw.submit_order(OrderRequest("AAPL", "BUY", 5.0, trade_date=date(2026, 1, 6), idempotency_key="b"), 100.0)
        self.assertEqual(o.status, OrderStatus.REJECTED)
        self.assertEqual(gw.ledger.cash, cash)
        self.assertEqual(gw.ledger.open_positions["AAPL"].shares, 5.0)

    def test_partial_sell_rejected(self):
        gw = self.make_gateway()
        gw.submit_order(OrderRequest("AAPL", "BUY", 10.0, trade_date=date(2026, 1, 5)), 100.0)
        cash = gw.ledger.cash
        o = gw.submit_order(OrderRequest("AAPL", "SELL", 4.0, trade_date=date(2026, 1, 6)), 100.0)
        self.assertEqual(o.status, OrderStatus.REJECTED)
        self.assertEqual(gw.ledger.cash, cash)

    def test_reconcile_detects_unexplained_cash_change(self):
        gw = self.make_gateway()
        gw.submit_order(OrderRequest("AAPL", "BUY", 5.0, trade_date=date(2026, 1, 5)), 100.0)
        self.assertTrue(gw.ledger.reconcile()[0])
        gw.ledger.cash += 50.0  # tamper
        ok, delta = gw.ledger.reconcile()
        self.assertFalse(ok)
        self.assertAlmostEqual(delta, 50.0, places=6)


class TestDurabilityAndRestart(_Base):
    def test_restart_replays_fills_and_blocks_duplicate(self):
        gw1 = self.make_gateway(with_store=True)
        req = OrderRequest("MSFT", "BUY", 3.0, trade_date=date(2026, 1, 5), idempotency_key="K1")
        o1 = gw1.submit_order(req, 300.0)
        self.assertEqual(o1.status, OrderStatus.FILLED)
        cash1 = gw1.ledger.cash

        # "Restart": brand-new gateway, fresh ledger, same durable store
        gw2 = self.make_gateway(with_store=True)
        self.assertAlmostEqual(gw2.ledger.cash, cash1, places=9)
        self.assertIn("MSFT", gw2.ledger.open_positions)
        o2 = gw2.submit_order(req, 300.0)
        self.assertEqual(o2.order_id, o1.order_id)
        self.assertAlmostEqual(gw2.ledger.cash, cash1, places=9)  # no second fill
        self.assertTrue(gw2.ledger.reconcile()[0])

    def test_state_history_persisted(self):
        gw = self.make_gateway(with_store=True)
        o = gw.submit_order(OrderRequest("AAPL", "BUY", 1.0, trade_date=date(2026, 1, 5)), 100.0)
        stored = OrderStore(self.db_path)
        self.addCleanup(stored.close)
        rows = stored.all_orders()
        self.assertEqual([h["status"] for h in rows[0]["state_history"]],
                         ["CREATED", "VALIDATED", "SUBMITTED", "FILLED"])
        self.assertEqual(rows[0]["order_id"], o.order_id)


class TestConcurrency(_Base):
    def test_concurrent_identical_submissions_fill_once(self):
        gw = self.make_gateway(with_store=True)
        req = OrderRequest("AAPL", "BUY", 5.0, trade_date=date(2026, 1, 5), idempotency_key="SAME")
        results = []

        def go():
            results.append(gw.submit_order(req, 100.0))

        threads = [threading.Thread(target=go) for _ in range(16)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        self.assertEqual(len({r.order_id for r in results}), 1)
        self.assertEqual(len(gw.ledger.transaction_history), 1)
        self.assertTrue(gw.ledger.reconcile()[0])

    def test_concurrent_distinct_orders_never_overspend(self):
        gw = self.make_gateway(capital=1000.0)
        tickers = [f"T{i}" for i in range(12)]

        def go(t):
            gw.submit_order(OrderRequest(t, "BUY", 2.0, trade_date=date(2026, 1, 5), idempotency_key=t), 100.0)

        threads = [threading.Thread(target=go, args=(t,)) for t in tickers]
        [t.start() for t in threads]
        [t.join() for t in threads]
        self.assertGreaterEqual(gw.ledger.cash, 0.0)
        self.assertTrue(gw.ledger.reconcile()[0])


if __name__ == "__main__":
    unittest.main()
