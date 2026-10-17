"""
Operational HTTP API for paper trading (Sprint 10.1, step F).

Dependency-free (stdlib http.server) because no web framework exists in this project;
the routing logic lives in `route()` which is a pure function and fully unit-tested.
This is NOT an order-entry API: there is no endpoint that submits or fills orders.

Security:
  * Binds to 127.0.0.1 by default.
  * Mutating endpoints (kill switch) require header `X-Admin-Token` equal to the
    TRADING_ADMIN_TOKEN environment variable; if the variable is unset they are disabled.
  * The token is never logged or returned.
Every request gets a correlation id (X-Correlation-ID honoured or generated), returned in
the response and included in structured log lines.

Run:  python -m monitoring.api  [--host 127.0.0.1 --port 8099]
"""

import argparse
import hmac
import json
import logging
import os
import re
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from monitoring.system_health import SystemHealthMonitor
from portfolio.risk_guard import RiskGuard

logger = logging.getLogger("OpsAPI")

TRADING_MODE = "PAPER_TRADING_ONLY"
BENCHMARK_REPORT = Path(__file__).resolve().parent.parent / "reports" / "sprint10" / "benchmark_metrics.json"


@dataclass
class ApiContext:
    risk_guard: RiskGuard
    monitor: SystemHealthMonitor
    orders_provider: Callable[[], List[Dict[str, Any]]]
    account_provider: Optional[Callable[[], Dict[str, Any]]] = None
    admin_token_env: str = "TRADING_ADMIN_TOKEN"
    benchmark_report: Path = BENCHMARK_REPORT


Response = Tuple[int, Dict[str, Any]]


def _authorized(ctx: ApiContext, headers: Dict[str, str]) -> Optional[Response]:
    expected = os.environ.get(ctx.admin_token_env, "")
    if not expected:
        return 403, {"error": "mutating endpoints disabled: admin token not configured"}
    supplied = headers.get("x-admin-token", "")
    if not hmac.compare_digest(supplied.encode(), expected.encode()):
        return 401, {"error": "invalid admin token"}
    return None


def route(ctx: ApiContext, method: str, path: str, headers: Dict[str, str],
          body: Optional[Dict[str, Any]], query: Optional[Dict[str, str]] = None) -> Response:
    """Pure request router. Returns (http_status, json_body)."""
    method = method.upper()
    query = query or {}
    headers = {k.lower(): v for k, v in headers.items()}

    if method == "GET":
        if path == "/health":
            return 200, ctx.monitor.get_health()
        if path == "/readiness":
            r = ctx.monitor.get_readiness()
            return (200 if r.get("ready") else 503), r
        if path == "/metrics":
            return 200, ctx.monitor.get_operational_metrics()
        if path == "/trading/status":
            active, state = ctx.risk_guard.is_kill_switch_active()
            out = {
                "trading_mode": TRADING_MODE,
                "live_trading_enabled": False,
                "kill_switch": {
                    "active": active, "reason": state.get("reason"),
                    "activated_at": state.get("activated_at"), "activated_by": state.get("activated_by"),
                },
                "accepting_new_orders": not active,
            }
            if ctx.account_provider:
                out["account"] = ctx.account_provider()
            return 200, out
        if path == "/trading/kill-switch/history":
            _, state = ctx.risk_guard.is_kill_switch_active()
            return 200, {"history": state.get("history", [])}
        if path == "/orders":
            orders = ctx.orders_provider()
            if "status" in query:
                orders = [o for o in orders if str(o.get("status")).upper() == query["status"].upper()]
            if "trace_id" in query:
                orders = [o for o in orders if o.get("trace_id") == query["trace_id"]]
            return 200, {"count": len(orders), "orders": orders}
        m = re.fullmatch(r"/orders/([A-Za-z0-9\-]+)", path)
        if m:
            for o in ctx.orders_provider():
                if o.get("order_id") == m.group(1):
                    return 200, o
            return 404, {"error": "order not found"}
        if path == "/audit":
            orders = ctx.orders_provider()
            _, state = ctx.risk_guard.is_kill_switch_active()
            events = [{"type": "ORDER_TRANSITION", "order_id": o["order_id"], "trace_id": o.get("trace_id"),
                       "ticker": o.get("ticker"), **h}
                      for o in orders for h in o.get("state_history", [])]
            events += [{"type": "KILL_SWITCH_" + h.get("action", "?"), **h} for h in state.get("history", [])]
            events.sort(key=lambda e: e.get("timestamp") or "")
            return 200, {"count": len(events), "events": events}
        if path == "/benchmark":
            if not ctx.benchmark_report.exists():
                return 404, {"status": "BLOCKED", "reason": "no benchmark report generated"}
            return 200, json.loads(ctx.benchmark_report.read_text(encoding="utf-8"))
        if path == "/api/pipeline/summary":
            summary_file = Path(__file__).resolve().parent.parent / "data" / "pipeline_summary.json"
            if summary_file.exists():
                try:
                    return 200, json.loads(summary_file.read_text(encoding="utf-8"))
                except Exception as e:
                    return 500, {"error": f"failed to load summary: {e}"}
            return 200, {
                "cycle_id": "NO_CYCLE_RUN_YET",
                "timestamp": None,
                "opportunities_screened": 0,
                "candidates_reviewed": 0,
                "orders_submitted": [],
                "orders_filled": [],
                "orders_rejected": [],
                "positions_closed": [],
                "top_opportunities": [],
                "committee_debates": []
            }
        return 404, {"error": "not found"}

    if method == "POST" and path == "/api/pipeline/run-cycle":
        denied = _authorized(ctx, headers)
        if denied:
            return denied
        try:
            from execution.pipeline_runner import TradingPipelineRunner
            runner = TradingPipelineRunner()
            summary = runner.run_cycle()
            return 200, summary
        except Exception as exc:
            logger.exception("Failed to run trading cycle via API")
            return 500, {"error": "cycle execution failed", "details": str(exc)}

    if method == "POST" and path in ("/trading/kill-switch/activate", "/trading/kill-switch/reactivate"):
        denied = _authorized(ctx, headers)
        if denied:
            return denied
        body = body or {}
        reason = str(body.get("reason", "")).strip()
        actor = str(body.get("actor", "")).strip()
        if not reason or not actor:
            return 400, {"error": "both 'reason' and 'actor' are required"}
        if path.endswith("/activate"):
            state = ctx.risk_guard.activate_kill_switch(reason, triggered_by=f"API:{actor}")
        else:
            state = ctx.risk_guard.deactivate_kill_switch(reason, approved_by=f"API:{actor}")
        return 200, {"active": state["active"], "reason": state["reason"]}

    return 404, {"error": "not found"}


def default_context() -> ApiContext:
    from execution.order_store import OrderStore  # local import keeps module import light
    store_path = Path(__file__).resolve().parent.parent / "data" / "paper_orders.sqlite"
    guard = RiskGuard()

    def provider() -> List[Dict[str, Any]]:
        if not store_path.exists():
            return []
        st = OrderStore(str(store_path))
        try:
            return st.all_orders()
        finally:
            st.close()

    return ApiContext(risk_guard=guard, monitor=SystemHealthMonitor(risk_guard=guard), orders_provider=provider)


def make_handler(ctx: ApiContext):
    class Handler(BaseHTTPRequestHandler):
        def _serve(self, method: str):
            from urllib.parse import parse_qsl, urlsplit
            parts = urlsplit(self.path)
            headers = {k.lower(): v for k, v in self.headers.items()}
            cid = headers.get("x-correlation-id") or str(uuid.uuid4())
            body = None
            length = int(self.headers.get("Content-Length") or 0)
            if length:
                try:
                    body = json.loads(self.rfile.read(min(length, 65536)))
                except json.JSONDecodeError:
                    self._send(400, {"error": "invalid JSON"}, cid)
                    return
            try:
                status, payload = route(ctx, method, parts.path, headers, body, dict(parse_qsl(parts.query)))
            except Exception as exc:  # noqa: BLE001 - never leak internals to the client
                logger.exception("handler error cid=%s", cid)
                status, payload = 500, {"error": "internal error", "type": type(exc).__name__}
            logger.info("api method=%s path=%s status=%s correlation_id=%s", method, parts.path, status, cid)
            self._send(status, payload, cid)

        def _send(self, status: int, payload: Dict[str, Any], cid: str):
            payload = {**payload, "correlation_id": cid}
            data = json.dumps(payload, default=str).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Correlation-ID", cid)
            self.end_headers()
            self.wfile.write(data)

        def _serve_static(self, rel_path: str):
            web_dir = Path(__file__).resolve().parent.parent / "web"
            file_path = (web_dir / rel_path).resolve()
            # Ensure path does not escape web_dir
            if not str(file_path).startswith(str(web_dir.resolve())):
                self.send_response(403)
                self.end_headers()
                return
            if not file_path.is_file():
                self.send_response(404)
                self.end_headers()
                return
            ext = file_path.suffix.lower()
            content_types = {
                ".html": "text/html; charset=utf-8",
                ".css": "text/css; charset=utf-8",
                ".js": "application/javascript; charset=utf-8",
                ".json": "application/json; charset=utf-8",
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".svg": "image/svg+xml",
                ".ico": "image/x-icon",
                ".woff2": "font/woff2",
                ".woff": "font/woff"
            }
            ctype = content_types.get(ext, "application/octet-stream")
            content = file_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            from urllib.parse import urlsplit
            parts = urlsplit(self.path)
            if parts.path in ("/", "/index.html"):
                self._serve_static("index.html")
                return
            if parts.path.startswith("/static/"):
                rel = parts.path[len("/static/"):]
                self._serve_static(f"static/{rel}")
                return
            self._serve("GET")

        def do_POST(self):
            self._serve("POST")

        def log_message(self, *args):  # silence default access log; structured log used instead
            pass

    return Handler


def serve(host: str = "127.0.0.1", port: int = 8099) -> None:
    server = ThreadingHTTPServer((host, port), make_handler(default_context()))
    logger.info("Ops API listening on %s:%d (mode=%s)", host, port, TRADING_MODE)
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8099)
    a = ap.parse_args()
    serve(a.host, a.port)
