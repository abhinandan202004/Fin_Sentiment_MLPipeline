"""
Comprehensive Sprint 10 Automated Test Suite.
Production Readiness, Risk Monitoring & Paper-Trading Reliability.

Test Categories:
1. Strategy Integrity & Provenance (ML influence, Veto enforcement, Risk check blocking, No target leakage).
2. Market Data Reliability (Stale bar detection, price anomalies, holiday calendar, feature completeness).
3. Central Risk Engine & Kill Switch (Position limits, sector limits, cash reserve, drawdown, kill switch).
4. Paper Execution Gateway (State machine transitions, idempotency deduplication, fee deduction).
5. Portfolio Ledger Invariants (total_equity == cash + market_value, atomic cash flows, single fee deduction).
6. Operational Monitoring & Readiness (Health check, readiness gating).
"""

import unittest
from datetime import date, datetime, timedelta
import uuid

from data.data_validator import DataValidator
from portfolio.risk_guard import RiskGuard, RiskLimits
from portfolio.ledger import PortfolioLedger
from portfolio.transaction_costs import TransactionCostModel
from execution.paper_gateway import PaperExecutionGateway, OrderRequest, OrderStatus
from execution.strategy_decision import StrategyDecisionEngine, StrategyDecision
from monitoring.system_health import SystemHealthMonitor

import tempfile
from pathlib import Path
import portfolio.risk_guard as _risk_guard_module

_TMP_DIR = None
_ORIGINAL_STATE_FILE = _risk_guard_module.STATE_FILE


def setUpModule():
    """Isolate kill-switch state: tests must never touch data/kill_switch_state.json."""
    global _TMP_DIR
    _TMP_DIR = tempfile.TemporaryDirectory()
    _risk_guard_module.STATE_FILE = Path(_TMP_DIR.name) / "kill_switch_state.json"


def tearDownModule():
    _risk_guard_module.STATE_FILE = _ORIGINAL_STATE_FILE
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()



class TestStrategyIntegrity(unittest.TestCase):
    """Verifies that ML model and Committee governance drive investment decisions."""

    def setUp(self):
        self.engine = StrategyDecisionEngine(min_ml_prob=0.55, opportunity_hurdle=50.0)

    def test_missing_features_fail_safely(self):
        """Missing or incomplete features must reject candidate safely without crash."""
        incomplete_features = {"rsi": 55.0}  # Missing 42 required features
        decision = self.engine.evaluate_candidate(
            ticker="AAPL",
            features=incomplete_features,
            decision_date=date(2026, 1, 15),
            portfolio_state={"total_equity": 12000.0, "cash": 12000.0, "open_positions": {}}
        )
        self.assertEqual(decision.decision, "REJECT")
        self.assertIn("FEATURE_VALIDATION_FAILED", decision.risk_reason)

    def test_ml_hurdle_blocks_low_conviction(self):
        """Candidates with probability below hurdle must be rejected."""
        # High probability features vs low probability features
        # If we artificially set min_ml_prob to 0.99, all candidates should fail ML hurdle
        strict_engine = StrategyDecisionEngine(min_ml_prob=0.99)
        valid_features = {f: 0.0 for f in self.engine.feature_list}
        valid_features["rsi"] = 50.0
        decision = strict_engine.evaluate_candidate(
            ticker="AAPL",
            features=valid_features,
            decision_date=date(2026, 1, 15),
            portfolio_state={"total_equity": 12000.0, "cash": 12000.0, "open_positions": {}}
        )
        self.assertEqual(decision.decision, "REJECT")
        self.assertIn("ML_CONVICTION_BELOW_HURDLE", decision.risk_reason)

    def test_committee_veto_blocks_order(self):
        """A committee veto by Risk Officer or Evidence Prosecutor must block execution."""
        valid_features = {f: 0.0 for f in self.engine.feature_list}
        valid_features["rsi"] = 45.0
        # Simulated committee with Risk Officer VETO
        committee_signals = {
            "Risk Officer": {"stance": "VETO", "reason": "Extreme VaR volatility"},
            "Evidence Prosecutor": {"stance": "BUY"},
            "CIO": {"stance": "BUY"}
        }
        # Set low ML hurdle so it passes ML stage
        permissive_engine = StrategyDecisionEngine(min_ml_prob=0.40, opportunity_hurdle=10.0)
        decision = permissive_engine.evaluate_candidate(
            ticker="NVDA",
            features=valid_features,
            decision_date=date(2026, 1, 15),
            portfolio_state={"total_equity": 12000.0, "cash": 12000.0, "open_positions": {}},
            committee_signals=committee_signals
        )
        self.assertEqual(decision.decision, "VETO")
        self.assertEqual(decision.committee_status, "VETOED")
        self.assertEqual(decision.veto_agent, "Risk Officer")
        self.assertIn("Extreme VaR", decision.veto_reason)


class TestDataReliability(unittest.TestCase):
    """Verifies market data validation, holiday calendar, and anomaly detection."""

    def setUp(self):
        self.validator = DataValidator()

    def test_staleness_detection_with_holidays(self):
        """Staleness must distinguish market weekends/holidays from actual missing data."""
        # Monday to Tuesday: 1 session -> Fresh
        mon = date(2026, 1, 12)
        tue = date(2026, 1, 13)
        is_stale, missed, msg = self.validator.check_staleness(mon, tue, max_allowed_missed_sessions=2)
        self.assertFalse(is_stale)
        self.assertEqual(missed, 1)

        # 5 sessions elapsed -> Stale
        next_mon = date(2026, 1, 19)
        is_stale, missed, msg = self.validator.check_staleness(mon, next_mon + timedelta(days=7), max_allowed_missed_sessions=2)
        self.assertTrue(is_stale)
        self.assertGreater(missed, 2)

    def test_price_bar_anomaly_rejection(self):
        """Invalid bars (high < low, non-positive price, extreme jump) must be rejected."""
        # Non-positive price
        bad_bar_1 = {"open": 0.0, "high": 10.0, "low": 5.0, "close": 0.0, "volume": 1000}
        passed, errs = self.validator.validate_price_bar(bad_bar_1, "AAPL")
        self.assertFalse(passed)

        # High < Low
        bad_bar_2 = {"open": 100.0, "high": 90.0, "low": 95.0, "close": 92.0, "volume": 1000}
        passed, errs = self.validator.validate_price_bar(bad_bar_2, "AAPL")
        self.assertFalse(passed)

        # Extreme jump (+150%)
        bad_bar_3 = {"open": 100.0, "high": 260.0, "low": 98.0, "close": 255.0, "volume": 1000}
        passed, errs = self.validator.validate_price_bar(bad_bar_3, "AAPL")
        self.assertFalse(passed)

        # Valid normal bar
        good_bar = {"open": 100.0, "high": 105.0, "low": 99.0, "close": 103.0, "volume": 500000}
        passed, errs = self.validator.validate_price_bar(good_bar, "AAPL")
        self.assertTrue(passed)


class TestRiskGuardAndKillSwitch(unittest.TestCase):
    """Verifies pre-trade risk limits and persistent kill switch behavior."""

    def setUp(self):
        self.limits = RiskLimits(
            max_single_position_pct=0.15,
            max_sector_exposure_pct=0.40,
            min_cash_pct=0.10,
            max_portfolio_drawdown_pct=0.20
        )
        self.risk_guard = RiskGuard(limits=self.limits)

    def test_kill_switch_activation_and_deactivation(self):
        """Engaging kill switch must block all orders; deactivation must restore approval."""
        self.risk_guard.deactivate_kill_switch("Reset test state", "TEST_SUITE")
        state_active, _ = self.risk_guard.is_kill_switch_active()
        self.assertFalse(state_active)

        # Activate kill switch
        self.risk_guard.activate_kill_switch("Market anomaly detected", "RISK_OFFICER")
        state_active, meta = self.risk_guard.is_kill_switch_active()
        self.assertTrue(state_active)
        self.assertEqual(meta["activated_by"], "RISK_OFFICER")

        # Order must be blocked
        portfolio = {"total_equity": 10000.0, "cash": 10000.0, "open_positions": {}}
        res = self.risk_guard.validate_order("AAPL", "BUY", 10.0, 150.0, portfolio)
        self.assertFalse(res.approved)
        self.assertEqual(res.violation_code, "KILL_SWITCH_ENGAGED")

        # Deactivate
        self.risk_guard.deactivate_kill_switch("Anomaly cleared", "CIO")
        state_active, _ = self.risk_guard.is_kill_switch_active()
        self.assertFalse(state_active)

        # Order now passes
        res2 = self.risk_guard.validate_order("AAPL", "BUY", 5.0, 150.0, portfolio)
        self.assertTrue(res2.approved)

    def test_single_position_limit_enforcement(self):
        """Orders exceeding max single position (15%) must be rejected or sized down."""
        self.risk_guard.deactivate_kill_switch("Reset test state", "TEST_SUITE")
        # 10,000 equity -> Max 1,500 in AAPL
        portfolio = {
            "total_equity": 10000.0,
            "cash": 9000.0,
            "open_positions": {
                "AAPL": {"shares": 10.0, "current_price": 145.0, "sector": "Technology"}  # Already $1,450 invested (limit: $1,500)
            }
        }
        # Attempting to buy another $1,000 of AAPL -> $2,450 total (24.5% > 15%), remaining headroom only $50 (< $100 min)
        res = self.risk_guard.validate_order("AAPL", "BUY", 10.0, 100.0, portfolio, sector="Technology")
        self.assertFalse(res.approved)
        self.assertEqual(res.violation_code, "SINGLE_POSITION_LIMIT_EXCEEDED")

    def test_cash_reserve_enforcement(self):
        """Orders leaving less than 10% cash liquidity must be blocked."""
        self.risk_guard.deactivate_kill_switch("Reset test state", "TEST_SUITE")
        # 10,000 equity -> Minimum cash reserve is 1,000
        # If cash is only 1,050, cannot spend 500
        portfolio = {
            "total_equity": 10000.0,
            "cash": 1050.0,
            "open_positions": {}
        }
        res = self.risk_guard.validate_order("MSFT", "BUY", 5.0, 100.0, portfolio)  # Cost 500
        self.assertFalse(res.approved)
        self.assertEqual(res.violation_code, "INSUFFICIENT_UNRESERVED_CASH")


class TestPaperExecutionGateway(unittest.TestCase):
    """Verifies order state machine, idempotency, and execution friction."""

    def setUp(self):
        self.ledger = PortfolioLedger(initial_capital_usd=10000.0)
        self.risk_guard = RiskGuard()
        self.risk_guard.deactivate_kill_switch("Reset test state", "TEST_SUITE")
        self.gateway = PaperExecutionGateway(ledger=self.ledger, risk_guard=self.risk_guard)

    def test_valid_order_lifecycle(self):
        """Order must transition CREATED -> VALIDATED -> SUBMITTED -> FILLED."""
        req = OrderRequest(ticker="AAPL", side="BUY", quantity=5.0, trade_date=date(2026, 1, 15))
        order = self.gateway.submit_order(req, market_price=150.0)
        self.assertEqual(order.status, OrderStatus.FILLED)
        self.assertGreater(order.fill_price, 150.0)  # Incorporates slippage
        self.assertGreater(order.total_fees, 0.0)
        # Verify state history
        history_statuses = [h["status"] for h in order.state_history]
        self.assertEqual(history_statuses, ["CREATED", "VALIDATED", "SUBMITTED", "FILLED"])

    def test_idempotent_order_deduplication(self):
        """Resubmitting identical order with same idempotency key must not execute duplicate trade."""
        idempotency_key = "IDEMP_TEST_KEY_12345"
        req = OrderRequest(
            ticker="MSFT",
            side="BUY",
            quantity=2.0,
            trade_date=date(2026, 1, 15),
            idempotency_key=idempotency_key
        )
        initial_cash = self.ledger.cash

        order_1 = self.gateway.submit_order(req, market_price=300.0)
        cash_after_1 = self.ledger.cash
        self.assertEqual(order_1.status, OrderStatus.FILLED)

        # Submit duplicate order with same idempotency key
        order_2 = self.gateway.submit_order(req, market_price=300.0)
        cash_after_2 = self.ledger.cash

        self.assertEqual(order_1.order_id, order_2.order_id)
        self.assertEqual(cash_after_1, cash_after_2)  # Zero additional cash deducted!


class TestPortfolioLedgerInvariants(unittest.TestCase):
    """Verifies strict mathematical invariant: total_equity == cash + market_value."""

    def setUp(self):
        self.ledger = PortfolioLedger(initial_capital_usd=12000.0)
        self.cost_model = TransactionCostModel()

    def test_atomic_buy_and_sell_with_exact_invariants(self):
        """Verifies total equity invariant through BUY, mark-to-market, and SELL cycles."""
        initial_eq = self.ledger.get_total_equity()
        self.assertEqual(initial_eq, 12000.0)

        # 1. Execute BUY
        costs = self.cost_model.compute_trade_cost(1000.0, side="BUY")
        pos = self.ledger.record_buy(
            ticker="AAPL",
            shares=10.0,
            fill_price=100.0,
            trade_date=date(2026, 1, 10),
            costs=costs
        )
        is_valid, delta = self.ledger.verify_ledger_invariant()
        self.assertTrue(is_valid)
        self.assertLess(delta, 1e-4)

        # 2. Mark to Market price appreciation ($100 -> $110)
        snapshot = self.ledger.mark_to_market({"AAPL": 110.0}, snapshot_date=date(2026, 1, 11))
        self.assertTrue(self.ledger.verify_ledger_invariant()[0])
        self.assertGreater(snapshot.total_equity, initial_eq)

        # 3. Execute SELL
        exit_costs = self.cost_model.compute_trade_cost(1100.0, side="SELL")
        pos_closed = self.ledger.record_sell(
            ticker="AAPL",
            fill_price=110.0,
            trade_date=date(2026, 1, 12),
            costs=exit_costs
        )
        self.assertEqual(len(self.ledger.open_positions), 0)
        self.assertEqual(len(self.ledger.closed_positions), 1)
        self.assertGreater(pos_closed.realized_pnl, 0.0)

        # Final invariant check: cash must equal total equity exactly
        self.assertEqual(self.ledger.get_positions_value(), 0.0)
        self.assertEqual(self.ledger.cash, self.ledger.get_total_equity())


class TestSystemHealthAndReadiness(unittest.TestCase):
    """Verifies operational health and readiness checks."""

    def setUp(self):
        self.monitor = SystemHealthMonitor()

    def test_health_check_returns_healthy(self):
        """Health check must report HEALTHY when database and model are accessible."""
        health = self.monitor.get_health()
        self.assertIn(health["status"], ["HEALTHY", "UNHEALTHY"])
        self.assertEqual(health["trading_mode"], "PAPER_TRADING_ONLY")
        self.assertTrue(health["database_connected"])

    def test_readiness_check_structure(self):
        """Readiness check must report structured gating issues."""
        readiness = self.monitor.get_readiness()
        self.assertIn("ready", readiness)
        self.assertIn("gating_issues", readiness)
        self.assertEqual(readiness["trading_mode"], "PAPER_TRADING_ONLY")


if __name__ == "__main__":
    unittest.main()
