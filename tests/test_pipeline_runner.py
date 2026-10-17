"""
Integration tests for TradingPipelineRunner (Autonomous daily trading cycle).
"""

import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

from execution.order_store import OrderStore
from execution.paper_gateway import PaperExecutionGateway
from execution.pipeline_runner import TradingPipelineRunner
from portfolio.ledger import PortfolioLedger
from portfolio.risk_guard import RiskGuard


class TestPipelineRunner(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.tmp_path = Path(self.tmp.name)

        # Isolated risk guard and store
        self.state_file = self.tmp_path / "kill.json"
        self.risk_guard = RiskGuard(state_file=self.state_file)
        self.order_db = self.tmp_path / "test_orders.sqlite"
        self.store = OrderStore(str(self.order_db))
        self.addCleanup(self.store.close)
        self.ledger = PortfolioLedger(initial_capital_usd=12000.0)

        self.gateway = PaperExecutionGateway(
            ledger=self.ledger,
            risk_guard=self.risk_guard,
            store=self.store
        )

        self.runner = TradingPipelineRunner(
            ledger=self.ledger,
            risk_guard=self.risk_guard,
            gateway=self.gateway,
            store=self.store
        )

    def test_runner_initialization(self):
        self.assertIsNotNone(self.runner.ledger)
        self.assertIsNotNone(self.runner.risk_guard)
        self.assertIsNotNone(self.runner.gateway)
        self.assertEqual(self.runner.ledger.cash, 12000.0)

    def test_run_cycle_dry_run(self):
        # Mock get_latest_market_data to be deterministic and fast
        def mock_market(ticker, target_date=None):
            return {
                "price": 150.0,
                "date": date(2026, 1, 15),
                "features": {
                    "rsi": 45.0,
                    "sentiment_score": 0.65,
                    "target": 1  # Note: strategy_decision engine ignores target
                }
            }

        with mock.patch.object(self.runner, "get_latest_market_data", side_effect=mock_market):
            summary = self.runner.run_cycle(
                universe=["AAPL", "MSFT", "GOOGL"],
                cycle_date=date(2026, 1, 15),
                top_k_candidates=2,
                execute_orders=False
            )

            self.assertEqual(summary["opportunities_screened"], 3)
            self.assertEqual(summary["candidates_reviewed"], 2)
            self.assertEqual(len(summary["orders_submitted"]), 0)
            self.assertTrue(summary["portfolio_after"]["ledger_reconciled"])
            self.assertAlmostEqual(summary["portfolio_after"]["reconcile_delta"], 0.0, places=6)

    def test_run_cycle_with_execution_and_reconciliation(self):
        def mock_market(ticker, target_date=None):
            return {
                "price": 100.0,
                "date": date(2026, 1, 15),
                "features": {
                    "rsi": 40.0,
                    "sentiment_score": 0.8,
                    "volatility_20d": 0.015
                }
            }

        with mock.patch.object(self.runner, "get_latest_market_data", side_effect=mock_market):
            summary = self.runner.run_cycle(
                universe=["AAPL", "NVDA"],
                cycle_date=date(2026, 1, 15),
                top_k_candidates=2,
                execute_orders=True
            )

            # Check cash reconciliation held strictly
            self.assertTrue(summary["portfolio_after"]["ledger_reconciled"])
            self.assertAlmostEqual(summary["portfolio_after"]["reconcile_delta"], 0.0, places=6)

            # Check orders in store
            stored_orders = self.store.all_orders()
            self.assertGreaterEqual(len(stored_orders), len(summary["orders_submitted"]))

    def test_run_cycle_blocks_when_kill_switch_active(self):
        self.risk_guard.activate_kill_switch("Integration test circuit breaker", triggered_by="Test")

        def mock_market(ticker, target_date=None):
            return {
                "price": 100.0,
                "date": date(2026, 1, 15),
                "features": {"rsi": 40.0, "sentiment_score": 0.8}
            }

        with mock.patch.object(self.runner, "get_latest_market_data", side_effect=mock_market):
            summary = self.runner.run_cycle(
                universe=["AAPL"],
                cycle_date=date(2026, 1, 15),
                top_k_candidates=1,
                execute_orders=True
            )

            # Any submitted order must be rejected by kill-switch
            if summary["orders_submitted"]:
                self.assertEqual(len(summary["orders_filled"]), 0)
                self.assertGreaterEqual(len(summary["orders_rejected"]), 1)


if __name__ == "__main__":
    unittest.main()
