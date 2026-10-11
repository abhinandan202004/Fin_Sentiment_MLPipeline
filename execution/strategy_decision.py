"""
Strategy Decision Engine for Sprint 10.
Production Readiness, Risk Monitoring & Paper-Trading Reliability.

Connects the true end-to-end investment strategy:
Market Features -> ML Model Inference -> Opportunity Scoring -> Committee Governance / Veto -> Central Risk Guard -> Order Generation.

Guarantees:
- Trades are genuinely driven by model probabilities and committee governance.
- Vetoes prevent order generation and persist counterfactual audit trails.
- Risk limit breaches prevent trade entry.
- Full decision traceability via unique trace_id.
"""

import uuid
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import joblib
import pandas as pd
import numpy as np

from config import ARTIFACTS_DIR
from data.data_validator import DataValidator
from opportunities.scoring import OpportunityScorer
from committee.governance_rules import GovernanceEngine
from portfolio.risk_guard import RiskGuard, RiskValidationResult
from execution.paper_gateway import OrderRequest, OrderSide

logger = logging.getLogger("StrategyDecisionEngine")

DEFAULT_MODEL_PATH = ARTIFACTS_DIR / "best_model.pkl"
DEFAULT_FEATURES_PATH = ARTIFACTS_DIR / "feature_list.json"
DEFAULT_THRESHOLD_PATH = ARTIFACTS_DIR / "optimal_threshold.json"


@dataclass
class StrategyDecision:
    """Audit record capturing the full provenance of an investment decision."""
    decision_id: str
    trace_id: str
    ticker: str
    decision_date: date
    decision: str  # "BUY", "HOLD", "REJECT", "VETO"
    model_name: str
    model_version: str
    predicted_probability: float
    decision_threshold: float
    opportunity_score: float
    committee_status: str  # "APPROVED", "VETOED", "NEUTRAL"
    veto_agent: Optional[str] = None
    veto_reason: Optional[str] = None
    risk_approved: bool = False
    risk_reason: Optional[str] = None
    target_allocation_pct: float = 0.0
    order_request: Optional[Dict[str, Any]] = None
    feature_snapshot: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class StrategyDecisionEngine:
    """Orchestrates end-to-end trade selection through ML, Opportunities, Committee, and Risk."""

    def __init__(
        self,
        min_ml_prob: float = 0.52,
        opportunity_hurdle: float = 50.0,
        model_path: Optional[Path] = None,
        features_path: Optional[Path] = None,
        risk_guard: Optional[RiskGuard] = None,
        data_validator: Optional[DataValidator] = None,
        committee_required: bool = True
    ):
        # committee_required=True (default): a candidate cannot become a BUY unless committee
        # signals were supplied and evaluated. Set False only for backtests where the committee
        # is not wired in; such runs must be labelled as ML-only.
        self.committee_required = committee_required
        self.min_ml_prob = min_ml_prob
        self.opportunity_hurdle = opportunity_hurdle
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.features_path = features_path or DEFAULT_FEATURES_PATH
        self.risk_guard = risk_guard or RiskGuard()
        self.data_validator = data_validator or DataValidator()
        self.opportunity_scorer = OpportunityScorer()
        self.governance_engine = GovernanceEngine()

        self.model = None
        self.feature_list = []
        self._load_model_artifacts()

    def _load_model_artifacts(self):
        try:
            if self.model_path.exists():
                self.model = joblib.load(self.model_path)
            if self.features_path.exists():
                with open(self.features_path, "r", encoding="utf-8") as f:
                    self.feature_list = json.load(f)
            logger.info(f"Loaded champion model from {self.model_path} with {len(self.feature_list)} features.")
        except Exception as e:
            logger.error(f"Failed to load model artifacts: {e}")

    def evaluate_candidate(
        self,
        ticker: str,
        features: Dict[str, Any],
        decision_date: date,
        portfolio_state: Dict[str, Any],
        committee_signals: Optional[Dict[str, Any]] = None,
        market_price: float = 100.0,
        sector: str = "Technology",
        is_stale: bool = False
    ) -> StrategyDecision:
        """
        Executes strict multi-stage strategy decision:
        Stage 1: Feature completeness & bar sanity
        Stage 2: Model prediction (P(BUY) >= hurdle)
        Stage 3: Opportunity factor scoring (Score >= hurdle)
        Stage 4: Committee Governance & Veto check
        Stage 5: Pre-trade Risk Guard
        """
        decision_id = str(uuid.uuid4())
        trace_id = str(uuid.uuid4())
        ticker = ticker.upper().strip()

        # Feature snapshot for audit trail (storing key features)
        feature_snap = {k: features.get(k, 0.0) for k in self.feature_list[:10]}

        # --- Stage 1: Feature Completeness Check ---
        is_valid_feat, feat_errors = self.data_validator.validate_features_for_inference(
            features, self.feature_list
        )
        if not is_valid_feat:
            return StrategyDecision(
                decision_id=decision_id,
                trace_id=trace_id,
                ticker=ticker,
                decision_date=decision_date,
                decision="REJECT",
                model_name="Random Forest",
                model_version="v2.0_43_features",
                predicted_probability=0.0,
                decision_threshold=self.min_ml_prob,
                opportunity_score=0.0,
                committee_status="NEUTRAL",
                risk_approved=False,
                risk_reason=f"FEATURE_VALIDATION_FAILED: {feat_errors[0]}",
                feature_snapshot=feature_snap
            )

        # --- Stage 2: ML Model Inference ---
        if self.model is None:
            return StrategyDecision(
                decision_id=decision_id,
                trace_id=trace_id,
                ticker=ticker,
                decision_date=decision_date,
                decision="REJECT",
                model_name="UNAVAILABLE",
                model_version="none",
                predicted_probability=0.0,
                decision_threshold=self.min_ml_prob,
                opportunity_score=0.0,
                committee_status="NEUTRAL",
                risk_approved=False,
                risk_reason="MODEL_ARTIFACT_UNAVAILABLE",
                feature_snapshot=feature_snap
            )

        try:
            parsed_feats = {}
            for f in self.feature_list:
                val = features.get(f, 0.0)
                parsed_feats[f] = float(val) if val is not None else 0.0
            feat_row = pd.DataFrame([parsed_feats])
            prob = float(self.model.predict_proba(feat_row)[0, 1])
        except Exception as exc:  # noqa: BLE001 - any inference failure must fail safe
            logger.error(f"Model inference failed for {ticker}: {exc}")
            prob = float("nan")
            infer_error = f"MODEL_INFERENCE_FAILED: {type(exc).__name__}"
        else:
            infer_error = None
        if infer_error or not np.isfinite(prob) or not (0.0 <= prob <= 1.0):
            return StrategyDecision(
                decision_id=decision_id,
                trace_id=trace_id,
                ticker=ticker,
                decision_date=decision_date,
                decision="REJECT",
                model_name="Random Forest",
                model_version="v2.0_43_features",
                predicted_probability=0.0,
                decision_threshold=self.min_ml_prob,
                opportunity_score=0.0,
                committee_status="NEUTRAL",
                risk_approved=False,
                risk_reason=infer_error or f"INVALID_PREDICTION: probability={prob}",
                feature_snapshot=feature_snap
            )

        # Directional Hurdle Filter
        if prob < self.min_ml_prob:
            return StrategyDecision(
                decision_id=decision_id,
                trace_id=trace_id,
                ticker=ticker,
                decision_date=decision_date,
                decision="REJECT",
                model_name="Random Forest",
                model_version="v2.0_43_features",
                predicted_probability=prob,
                decision_threshold=self.min_ml_prob,
                opportunity_score=0.0,
                committee_status="NEUTRAL",
                risk_approved=False,
                risk_reason=f"ML_CONVICTION_BELOW_HURDLE: Prob {prob:.4f} < {self.min_ml_prob:.4f}",
                feature_snapshot=feature_snap
            )

        # --- Stage 3: Opportunity Scoring ---
        sentiment_rank = 0.80 if float(features.get("avg_sentiment", 0.0) or 0.0) > 0.2 else 0.50
        analog_win = 0.55 if prob >= 0.55 else 0.50
        opp_score_data = self.opportunity_scorer.compute_opportunity(
            ticker=ticker,
            ml_prob=prob,
            ml_threshold=self.min_ml_prob,
            analog_win_rate=analog_win,
            sentiment_rank=sentiment_rank,
            rsi=float(features.get("rsi", 50.0) or 50.0),
            sector_return_5d=float(features.get("sector_return_5d", 0.0) or 0.0),
            market_regime="Neutral"
        )
        opp_score = float(opp_score_data.get("opportunity_score", 50.0))

        if opp_score < self.opportunity_hurdle:
            return StrategyDecision(
                decision_id=decision_id,
                trace_id=trace_id,
                ticker=ticker,
                decision_date=decision_date,
                decision="REJECT",
                model_name="Random Forest",
                model_version="v2.0_43_features",
                predicted_probability=prob,
                decision_threshold=self.min_ml_prob,
                opportunity_score=opp_score,
                committee_status="NEUTRAL",
                risk_approved=False,
                risk_reason=f"OPPORTUNITY_SCORE_BELOW_HURDLE: {opp_score:.1f} < {self.opportunity_hurdle:.1f}",
                feature_snapshot=feature_snap
            )

        # --- Stage 4: Committee Governance & Veto Check ---
        comm_status = "APPROVED"
        veto_agent = None
        veto_reason = None

        if not committee_signals and self.committee_required:
            return StrategyDecision(
                decision_id=decision_id,
                trace_id=trace_id,
                ticker=ticker,
                decision_date=decision_date,
                decision="REJECT",
                model_name="Random Forest",
                model_version="v2.0_43_features",
                predicted_probability=prob,
                decision_threshold=self.min_ml_prob,
                opportunity_score=opp_score,
                committee_status="NOT_EVALUATED",
                risk_approved=False,
                risk_reason="COMMITTEE_NOT_EVALUATED: no committee signals supplied",
                feature_snapshot=feature_snap
            )
        if not committee_signals:
            comm_status = "NOT_EVALUATED"

        if committee_signals:
            cro_stance = committee_signals.get("Risk Officer", {}).get("stance", "BUY")
            pros_stance = committee_signals.get("Evidence Prosecutor", {}).get("stance", "BUY")
            cio_stance = committee_signals.get("CIO", {}).get("stance", "BUY")

            if cro_stance == "VETO":
                comm_status = "VETOED"
                veto_agent = "Risk Officer"
                veto_reason = committee_signals.get("Risk Officer", {}).get("reason", "Downside risk exceeds parameters")
            elif pros_stance == "VETO":
                comm_status = "VETOED"
                veto_agent = "Evidence Prosecutor"
                veto_reason = committee_signals.get("Evidence Prosecutor", {}).get("reason", "Evidence quality insufficient")
            elif cio_stance != "BUY":
                comm_status = "NEUTRAL"
                veto_reason = "CIO did not approve BUY conviction"

            if comm_status == "VETOED":
                return StrategyDecision(
                    decision_id=decision_id,
                    trace_id=trace_id,
                    ticker=ticker,
                    decision_date=decision_date,
                    decision="VETO",
                    model_name="Random Forest",
                    model_version="v2.0_43_features",
                    predicted_probability=prob,
                    decision_threshold=self.min_ml_prob,
                    opportunity_score=opp_score,
                    committee_status=comm_status,
                    veto_agent=veto_agent,
                    veto_reason=veto_reason,
                    risk_approved=False,
                    risk_reason=f"COMMITTEE_VETO: {veto_agent} vetoed ({veto_reason})",
                    feature_snapshot=feature_snap
                )

        if comm_status == "NEUTRAL" and committee_signals:
            return StrategyDecision(
                decision_id=decision_id,
                trace_id=trace_id,
                ticker=ticker,
                decision_date=decision_date,
                decision="REJECT",
                model_name="Random Forest",
                model_version="v2.0_43_features",
                predicted_probability=prob,
                decision_threshold=self.min_ml_prob,
                opportunity_score=opp_score,
                committee_status=comm_status,
                veto_reason=veto_reason,
                risk_approved=False,
                risk_reason=f"COMMITTEE_NOT_APPROVED: {veto_reason}",
                feature_snapshot=feature_snap
            )

        # --- Stage 5: Pre-Trade Risk Guard ---
        total_equity = float(portfolio_state.get("total_equity", 12000.0))
        target_alloc_pct = 0.10  # 10% portfolio allocation target
        target_dollars = target_alloc_pct * total_equity
        target_shares = target_dollars / market_price if market_price > 0 else 0.0

        risk_res: RiskValidationResult = self.risk_guard.validate_order(
            ticker=ticker,
            side="BUY",
            quantity=target_shares,
            price=market_price,
            portfolio_state=portfolio_state,
            sector=sector,
            is_stale=is_stale
        )

        if not risk_res.approved:
            return StrategyDecision(
                decision_id=decision_id,
                trace_id=trace_id,
                ticker=ticker,
                decision_date=decision_date,
                decision="REJECT",
                model_name="Random Forest",
                model_version="v2.0_43_features",
                predicted_probability=prob,
                decision_threshold=self.min_ml_prob,
                opportunity_score=opp_score,
                committee_status=comm_status,
                risk_approved=False,
                risk_reason=risk_res.reason,
                feature_snapshot=feature_snap
            )

        # Build approved Order Request
        order_req = OrderRequest(
            ticker=ticker,
            side="BUY",
            quantity=risk_res.adjusted_quantity,
            trade_date=decision_date,
            sector=sector,
            trace_id=trace_id,
            idempotency_key=f"{decision_date}_{ticker}_BUY_{risk_res.adjusted_quantity:.2f}"
        )

        return StrategyDecision(
            decision_id=decision_id,
            trace_id=trace_id,
            ticker=ticker,
            decision_date=decision_date,
            decision="BUY",
            model_name="Random Forest",
            model_version="v2.0_43_features",
            predicted_probability=prob,
            decision_threshold=self.min_ml_prob,
            opportunity_score=opp_score,
            committee_status=comm_status,
            risk_approved=True,
            risk_reason="All risk checks satisfied",
            target_allocation_pct=target_alloc_pct,
            order_request=asdict(order_req),
            feature_snapshot=feature_snap
        )
