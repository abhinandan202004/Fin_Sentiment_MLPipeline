"""
Tests for Command Center Web Dashboard static serving and API endpoints.
"""

import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from monitoring.api import ApiContext, make_handler
from portfolio.risk_guard import RiskGuard


class _DummyMonitor:
    def get_health(self):
        return {"status": "HEALTHY"}

    def get_readiness(self):
        return {"ready": True, "gating_issues": []}

    def get_operational_metrics(self):
        return {"status": "OK"}


class TestWebDashboardServing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.guard = RiskGuard(state_file=Path(cls.tmp.name) / "kill.json")
        cls.ctx = ApiContext(
            risk_guard=cls.guard,
            monitor=_DummyMonitor(),
            orders_provider=lambda: [
                {"order_id": "test-123", "status": "FILLED", "ticker": "AAPL", "side": "BUY", "quantity": 10.0}
            ]
        )
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(cls.ctx))
        cls.port = cls.server.server_address[1]
        cls.base_url = f"http://127.0.0.1:{cls.port}"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.tmp.cleanup()

    def test_serve_index_html(self):
        req = urllib.request.Request(f"{self.base_url}/")
        with urllib.request.urlopen(req, timeout=5) as resp:
            self.assertEqual(resp.status, 200)
            self.assertIn("text/html", resp.headers.get("Content-Type", ""))
            content = resp.read().decode("utf-8")
            self.assertIn("FIN_SENTIMENT", content)
            self.assertIn("PAPER_TRADING_ONLY", content)
            self.assertIn("Plus+Jakarta+Sans", content)

    def test_serve_dashboard_css(self):
        req = urllib.request.Request(f"{self.base_url}/static/css/dashboard.css")
        with urllib.request.urlopen(req, timeout=5) as resp:
            self.assertEqual(resp.status, 200)
            self.assertIn("text/css", resp.headers.get("Content-Type", ""))
            content = resp.read().decode("utf-8")
            self.assertIn("#051F20", content)
            self.assertIn("#DAF1DE", content)
            self.assertIn("#A3E635", content)

    def test_serve_dashboard_js(self):
        req = urllib.request.Request(f"{self.base_url}/static/js/dashboard.js")
        with urllib.request.urlopen(req, timeout=5) as resp:
            self.assertEqual(resp.status, 200)
            self.assertIn("javascript", resp.headers.get("Content-Type", ""))
            content = resp.read().decode("utf-8")
            self.assertIn("loadSystemHealth", content)
            self.assertIn("switchTab", content)

    def test_pipeline_summary_endpoint(self):
        req = urllib.request.Request(f"{self.base_url}/api/pipeline/summary")
        with urllib.request.urlopen(req, timeout=5) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertIn("opportunities_screened", data)
            self.assertIn("correlation_id", data)

    def test_path_traversal_blocked(self):
        # Escaping web directory must be 403 or 404
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(f"{self.base_url}/static/../../config.py", timeout=5)
        self.assertIn(cm.exception.code, [403, 404])


if __name__ == "__main__":
    unittest.main()
