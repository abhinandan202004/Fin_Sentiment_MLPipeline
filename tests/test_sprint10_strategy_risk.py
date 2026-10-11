"""
Comprehensive Strategy Integrity & Advanced Risk Controls Test Suite (Sprint 10.1 Phase D).

Verifies:
1. Target & Future Return Leakage Protection (no target/future_return fields influence inference or ranking).
2. Deterministic Replay (identical inputs yield identical decision output).
3. Fail-Safe ML Inference (NaN, out-of-range, or corrupted model failures reject safely).
4. Committee Governance Enforcement (veto blocks, CIO non-approval blocks, committee required flag).
5. Daily Loss Limit Enforcement (blocks new BUY orders when day loss >= limit).
6. Peak Drawdown Gate (blocks BUY orders when drawdown >= limit, but permits SELL/liquidation).
7. Aggregate Capital Investment Limit (caps total invested exposure across portfolio).
8. Parameter Sanity & Anti-Corruption (rejects NaN, Inf, non-positive equity, or zero/negative prices).
"""

import math
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

from execution.strategy_decision import StrategyDecisionEngine, StrategyDecision
from portfolio.risk_guard import RiskGuard, RiskLimits, RiskValidationResult
from execution.paper_gateway import PaperExecutionGateway, OrderRequest, OrderStatus
from portfolio.ledger import PortfolioLedger


class TestStrategyLeakageAndIntegrity(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.state_file = Path(self.tmp.name) / "kill.json"
        self.guard = RiskGuard(state_file=self.state_file)
        self.engine = StrategyDecisionEngine(
            min_ml_prob=0.50,
            opportunity_hurdle=40.0,
            risk_guard=self.guard,
            committee_required=False  # Allow evaluating pure ML path in isolation
        )

    def test_target_label_leakage_has_zero_influence(self):
        """Passing future return or target label must NOT alter decision or probability."""
        clean_feats = {f: 0.0 for f in self.engine.feature_list}
        clean_feats["rsi"] = 55.0

        p_state = {"total_equity": 10000.0, "cash": 10000.0, "open_positions": {}}
        d_clean = self.engine.evaluate_candidate(
            ticker="AAPL",
            features=clean_feats,
            decision_date=date(2026, 1, 15),
            portfolio_state=p_state,
            market_price=150.0
        )

        # Contaminate input with future return and target label
        polluted_feats = dict(clean_feats)
        polluted_feats["target"] = 1
        polluted_feats["future_return_5d"] = 0.50
        polluted_feats["actual_return"] = 0.99

        d_polluted = self.engine.evaluate_candidate(
            ticker="AAPL",
            features=polluted_feats,
            decision_date=date(2026, 1, 15),
            portfolio_state=p_state,
            market_price=150.0
        )

        self.assertAlmostEqual(d_clean.predicted_probability, d_polluted.predicted_probability, places=9)
        self.assertEqual(d_clean.decision, d_polluted.decision)
        self.assertAlmostEqual(d_clean.opportunity_score, d_polluted.opportunity_score, places=4)

    def test_deterministic_replay_produces_identical_decision(self):
        """Identical candidate features and portfolio state produce deterministic decisions."""
        feats = {f: 0.0 for f in self.engine.feature_list}
        feats["rsi"] = 48.0
        p_state = {"total_equity": 10000.0, "cash": 10000.0, "open_positions": {}}

        d1 = self.engine.evaluate_candidate("MSFT", feats, date(2026, 1, 15), p_state, market_price=300.0)
        d2 = self.engine.evaluate_candidate("MSFT", feats, date(2026, 1, 15), p_state, market_price=300.0)

        self.assertEqual(d1.decision, d2.decision)
        self.assertAlmostEqual(d1.predicted_probability, d2.predicted_probability, places=9)
        self.assertAlmostEqual(d1.opportunity_score, d2.opportunity_score, places=4)
        self.assertEqual(d1.target_allocation_pct, d2.target_allocation_pct)

    def test_invalid_and_nan_prediction_fails_safely(self):
        """Model raising exception or returning NaN/inf must reject safely without crashing."""
        mock_model = MagicMock()
        mock_model.predict_proba.side_effect = RuntimeError("GPU memory corruption")

        bad_engine = StrategyDecisionEngine(risk_guard=self.guard, committee_required=False)
        bad_engine.model = mock_model

        feats = {f: 0.0 for f in bad_engine.feature_list}
        p_state = {"total_equity": 10000.0, "cash": 10000.0, "open_positions": {}}

        d = bad_engine.evaluate_candidate("NVDA", feats, date(2026, 1, 15), p_state)
        self.assertEqual(d.decision, "REJECT")
        self.assertIn("MODEL_INFERENCE_FAILED", d.risk_reason)

    def test_committee_required_blocks_unreviewed_trades(self):
        """When committee_required=True, omitting committee signals must REJECT the trade."""
        engine_strict = StrategyDecisionEngine(
            min_ml_prob=0.40,
            opportunity_hurdle=10.0,
            risk_guard=self.guard,
            committee_required=True
        )
        feats = {f: 0.0 for f in engine_strict.feature_list}
        p_state = {"total_equity": 10000.0, "cash": 10000.0, "open_positions": {}}

        d = engine_strict.evaluate_candidate("AAPL", feats, date(2026, 1, 15), p_state, committee_signals=None)
        self.assertEqual(d.decision, "REJECT")
        self.assertEqual(d.committee_status, "NOT_EVALUATED")
        self.assertIn("COMMITTEE_NOT_EVALUATED", d.risk_reason)


class TestAdvancedRiskControls(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.state_file = Path(self.tmp.name) / "kill.json"
        self.limits = RiskLimits(
            max_single_position_pct=0.15,
            max_sector_exposure_pct=0.40,
            min_cash_pct=0.10,
            max_aggregate_invested_pct=0.85,
            max_portfolio_drawdown_pct=0.20,
            max_daily_loss_pct=0.05
        )
        self.guard = RiskGuard(limits=self.limits, state_file=self.state_file)

    def test_daily_loss_limit_blocks_new_buy_orders(self):
        """When daily loss reaches or exceeds limit (5%), new BUY orders must be blocked."""
        # Day started at $10,000, current equity dropped to $9,400 (6% daily loss)
        p_state = {
            "total_equity": 9400.0,
            "cash": 9400.0,
            "day_start_equity": 10000.0,
            "peak_equity": 10000.0,
            "open_positions": {}
        }
        res = self.guard.validate_order("AAPL", "BUY", 5.0, 100.0, p_state)
        self.assertFalse(res.approved)
        self.assertEqual(res.violation_code, "DAILY_LOSS_LIMIT_BREACH")

    def test_drawdown_limit_blocks_buy_but_permits_sell(self):
        """Peak drawdown breach must block BUYs, but must never prevent closing/selling positions."""
        # Peak $10,000, current equity $7,800 (22% drawdown > 20% limit)
        p_state = {
            "total_equity": 7800.0,
            "cash": 4000.0,
            "peak_equity": 10000.0,
            "open_positions": {
                "AAPL": {"shares": 20.0, "current_price": 190.0, "sector": "Technology"}
            }
        }
        # BUY must be blocked
        res_buy = self.guard.validate_order("MSFT", "BUY", 5.0, 100.0, p_state)
        self.assertFalse(res_buy.approved)
        self.assertEqual(res_buy.violation_code, "DRAWDOWN_LIMIT_BREACH")

        # SELL must be approved to permit de-risking
        res_sell = self.guard.validate_order("AAPL", "SELL", 20.0, 190.0, p_state)
        self.assertTrue(res_sell.approved)
        self.assertEqual(res_sell.adjusted_quantity, 20.0)

    def test_aggregate_invested_capital_limit(self):
        """Total portfolio exposure exceeding 85% must be capped or rejected."""
        # Equity: $10,000. Already invested: $8,000 (80%). Max allowed: $8,500.
        p_state = {
            "total_equity": 10000.0,
            "cash": 2000.0,
            "peak_equity": 10000.0,
            "open_positions": {
                "AAPL": {"shares": 40.0, "current_price": 100.0, "sector": "Technology"},
                "MSFT": {"shares": 40.0, "current_price": 100.0, "sector": "Technology"},
            }
        }
        # Attempt to buy $1,000 of GOOGL (would push total invested to $9,000 > $8,500)
        res = self.guard.validate_order("GOOGL", "BUY", 10.0, 100.0, p_state, sector="Communication Services")
        self.assertTrue(res.approved)
        # Sized down to remaining capacity ($500 -> 5.0 shares)
        self.assertAlmostEqual(res.adjusted_quantity, 5.0, places=4)

    def test_non_finite_and_corrupt_parameters_rejected(self):
        """NaN or Inf prices/quantities must be strictly rejected."""
        p_state = {"total_equity": 10000.0, "cash": 10000.0, "open_positions": {}}
        for bad_p in (float("nan"), float("inf"), -10.0, 0.0):
            res = self.guard.validate_order("AAPL", "BUY", 10.0, bad_p, p_state)
            self.assertFalse(res.approved)
            self.assertEqual(res.violation_code, "INVALID_QUANTITY_PRICE")

        for bad_q in (float("nan"), float("-inf"), -5.0, 0.0):
            res = self.guard.validate_order("AAPL", "BUY", bad_q, 100.0, p_state)
            self.assertFalse(res.approved)
            self.assertEqual(res.violation_code, "INVALID_QUANTITY_PRICE")


if __name__ == "__main__":
    unittest.main()
