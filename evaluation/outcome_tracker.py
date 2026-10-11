"""
Unified Outcome Tracker for Sprint 9: Autonomous Evaluation & Adaptive Intelligence.

Provides zero-look-ahead market outcome resolution across multiple forward horizons:
  - 1-day (daily persistence)
  - 5-day (tactical swing horizon, primary ML target)
  - 10-day (fortnightly horizon)
  - 20-day (monthly trend horizon)
  - 60-day (quarterly fundamental horizon)

Resolves realized forward market outcomes for:
  1. ML Predictions (populating PredictionOutcome)
  2. Agent Decisions (AgentDecisionRecord)
  3. Investment Committee Decisions (CommitteeDecision / CommitteeEvaluationRecord)
  4. Discovered Opportunities (OpportunityOutcome)

Strictly adheres to:
  - Rule 1: No Look-Ahead Bias (only uses price bars strictly on or after decision date)
  - Rule 2: Multi-Horizon market session sequencing (handles weekends, holidays, missing bars)
"""

import sys
import logging
from bisect import bisect_left
from datetime import datetime, date, timedelta, timezone
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np

from database.connection import SessionLocal, init_db
from database.models import (
    MarketData,
    Prediction,
    PredictionOutcome,
    AgentDecisionRecord,
    CommitteeDecision,
    OpportunityHistory,
    OpportunityOutcome,
    Company
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("OutcomeTracker")


class OutcomeTracker:
    """
    High-performance outcome resolver backed by database market bars
    with automatic in-memory indexing to prevent look-ahead bias.
    """

    def __init__(self, session=None):
        self._external_session = session
        self._bars_cache: Dict[str, List[MarketData]] = {}
        self._dates_cache: Dict[str, List[date]] = {}
        self._sectors_cache: Dict[str, str] = {}

    def _get_session(self):
        return self._external_session or SessionLocal()

    def preload_market_bars(self, tickers: Optional[List[str]] = None) -> None:
        """
        Preloads market data bars into memory sorted by date ascending.
        This provides O(log N) session index lookups without redundant DB queries.
        """
        session = self._get_session()
        should_close = self._external_session is None
        try:
            query = session.query(MarketData).order_by(MarketData.ticker, MarketData.date.asc())
            if tickers:
                query = query.filter(MarketData.ticker.in_(tickers))
            else:
                self._bars_cache.clear()
                self._dates_cache.clear()
            bars = query.all()

            for b in bars:
                t = b.ticker
                if t not in self._bars_cache:
                    self._bars_cache[t] = []
                    self._dates_cache[t] = []
                self._bars_cache[t].append(b)
                self._dates_cache[t].append(b.date)
                
            # Preload company sectors
            companies = session.query(Company).all()
            self._sectors_cache = {c.ticker: (c.sector or "Unknown") for c in companies}
            
            logger.info(f"Loaded market bars for {len(self._bars_cache)} tickers ({len(bars)} bars queried).")
        finally:
            if should_close:
                session.close()

    def get_forward_outcomes(self, ticker: str, decision_date: Union[date, datetime]) -> Optional[Dict[str, Any]]:
        """
        Calculates 1d, 5d, 10d, 20d, and 60d forward returns strictly starting from
        the first trading session on or after decision_date (Entry Bar).
        Tracks each horizon independently with explicit status: PENDING, RESOLVED, UNRESOLVABLE.
        """
        if not self._bars_cache:
            self.preload_market_bars()

        def _empty_outcome(status: str, reason: str) -> Dict[str, Any]:
            return {
                "status": status,
                "reason": reason,
                "entry_date": None,
                "entry_price": None,
                "exit_price_5d": None,
                "available_forward_sessions": 0,
                "return_1d": None,
                "return_5d": None,
                "return_10d": None,
                "return_20d": None,
                "return_60d": None,
                "horizon_status": {h: status for h in ["1d", "5d", "10d", "20d", "60d"]},
                "max_drawdown_20d": None,
                "actual_direction": None,
            }

        if ticker not in self._bars_cache:
            return _empty_outcome("UNRESOLVABLE", f"Ticker {ticker} not found in market cache")

        bars = self._bars_cache[ticker]
        dates = self._dates_cache[ticker]
        if not bars:
            return _empty_outcome("UNRESOLVABLE", "No price bars available")

        d_date = decision_date.date() if isinstance(decision_date, datetime) else decision_date

        entry_idx = bisect_left(dates, d_date)
        if entry_idx >= len(bars):
            # Decision date is after our latest available bar (future/pending)
            return _empty_outcome("PENDING", "Awaiting market bars after decision date")

        entry_bar = bars[entry_idx]
        entry_price = float(entry_bar.close)
        if entry_price <= 0.0:
            return _empty_outcome("UNRESOLVABLE", "Non-positive entry price")

        total_forward_sessions = len(bars) - 1 - entry_idx

        horizon_steps = {"1d": 1, "5d": 5, "10d": 10, "20d": 20, "60d": 60}
        horizon_returns = {}
        horizon_status = {}
        exit_prices = {}

        for h_name, step in horizon_steps.items():
            if total_forward_sessions >= step:
                p_fwd = float(bars[entry_idx + step].close)
                horizon_returns[h_name] = round((p_fwd - entry_price) / entry_price, 5)
                horizon_status[h_name] = "RESOLVED"
                exit_prices[h_name] = p_fwd
            else:
                horizon_returns[h_name] = None
                horizon_status[h_name] = "PENDING"
                exit_prices[h_name] = None

        # Max drawdown across forward window up to 20 sessions (never treating missing as 0)
        window_bars = bars[entry_idx + 1: entry_idx + min(21, total_forward_sessions + 1)]
        if window_bars:
            min_low = min(float(b.low) for b in window_bars)
            max_dd = round((min_low - entry_price) / entry_price, 5)
        else:
            max_dd = None

        ret_5d = horizon_returns["5d"]
        actual_dir = "BUY" if (ret_5d is not None and ret_5d > 0) else ("NOT BUY" if ret_5d is not None else None)

        overall_status = "RESOLVED" if horizon_status["5d"] == "RESOLVED" else ("RESOLVED" if any(s == "RESOLVED" for s in horizon_status.values()) else "PENDING")

        return {
            "status": overall_status,
            "entry_date": entry_bar.date,
            "entry_price": entry_price,
            "exit_price_5d": exit_prices["5d"],
            "available_forward_sessions": total_forward_sessions,
            "return_1d": horizon_returns["1d"],
            "return_5d": horizon_returns["5d"],
            "return_10d": horizon_returns["10d"],
            "return_20d": horizon_returns["20d"],
            "return_60d": horizon_returns["60d"],
            "horizon_status": horizon_status,
            "max_drawdown_20d": max_dd,
            "actual_direction": actual_dir
        }

    def resolve_prediction_outcomes(self, limit: Optional[int] = None, force_recompute: bool = False) -> int:
        """
        Idempotently resolves pending Prediction records into PredictionOutcome records.
        """
        session = self._get_session()
        should_close = self._external_session is None
        try:
            if not self._bars_cache:
                self.preload_market_bars()

            # Existing records map
            existing_outcomes = {
                r.prediction_id: r for r in session.query(PredictionOutcome).filter(PredictionOutcome.prediction_id.isnot(None)).all()
            }

            preds_query = session.query(Prediction)
            if limit:
                preds_query = preds_query.limit(limit)

            pending_preds = preds_query.all()
            resolved_count = 0

            for pred in pending_preds:
                existing = existing_outcomes.get(pred.id)
                if existing and not force_recompute and existing.status == "RESOLVED" and existing.actual_return_60d is not None:
                    continue  # Fully resolved already

                outcomes = self.get_forward_outcomes(pred.ticker, pred.generated_at)
                if not outcomes or outcomes["status"] == "UNRESOLVABLE":
                    continue

                threshold = 0.60
                pred_prob = float(pred.confidence)
                pred_dir = "BUY" if pred_prob >= threshold else "NOT BUY"

                r1 = outcomes.get("return_1d")
                r5 = outcomes.get("return_5d")
                r10 = outcomes.get("return_10d")
                r20 = outcomes.get("return_20d")
                r60 = outcomes.get("return_60d")

                c1 = ((r1 > 0) == (pred_dir == "BUY")) if r1 is not None else None
                c5 = ((r5 > 0) == (pred_dir == "BUY")) if r5 is not None else None
                c10 = ((r10 > 0) == (pred_dir == "BUY")) if r10 is not None else None
                c20 = ((r20 > 0) == (pred_dir == "BUY")) if r20 is not None else None
                c60 = ((r60 > 0) == (pred_dir == "BUY")) if r60 is not None else None

                y_true = 1.0 if (r5 is not None and r5 > 0) else (0.0 if r5 is not None else None)
                brier = round(float((pred_prob - y_true) ** 2), 5) if y_true is not None else None
                sector = self._sectors_cache.get(pred.ticker, "Technology")

                if existing:
                    # Idempotent update
                    existing.status = outcomes["status"]
                    existing.entry_price = outcomes.get("entry_price")
                    existing.exit_price_5d = outcomes.get("exit_price_5d")
                    existing.max_drawdown = outcomes.get("max_drawdown_20d")
                    existing.horizon_status = outcomes.get("horizon_status")
                    existing.actual_return_1d = r1
                    existing.actual_return_5d = r5
                    existing.actual_return_10d = r10
                    existing.actual_return_20d = r20
                    existing.actual_return_60d = r60
                    existing.actual_direction = outcomes.get("actual_direction")
                    existing.correct_1d = c1
                    existing.correct_5d = c5
                    existing.correct_10d = c10
                    existing.correct_20d = c20
                    existing.correct_60d = c60
                    existing.brier_score = brier
                    existing.resolved_date = datetime.now(timezone.utc)
                else:
                    po = PredictionOutcome(
                        prediction_id=pred.id,
                        ticker=pred.ticker,
                        prediction_date=pred.generated_at,
                        predicted_probability=pred_prob,
                        threshold=threshold,
                        predicted_direction=pred_dir,
                        status=outcomes["status"],
                        entry_price=outcomes.get("entry_price"),
                        exit_price_5d=outcomes.get("exit_price_5d"),
                        max_drawdown=outcomes.get("max_drawdown_20d"),
                        horizon_status=outcomes.get("horizon_status"),
                        actual_return_1d=r1,
                        actual_return_5d=r5,
                        actual_return_10d=r10,
                        actual_return_20d=r20,
                        actual_return_60d=r60,
                        actual_direction=outcomes.get("actual_direction"),
                        correct_1d=c1,
                        correct_5d=c5,
                        correct_10d=c10,
                        correct_20d=c20,
                        correct_60d=c60,
                        brier_score=brier,
                        market_regime="Bull",
                        sector=sector,
                        model_version=pred.model_version or "SPRINT_9_BASELINE",
                        feature_version="v2.0_43_features",
                        threshold_version="calibrated_0.60",
                        calibration_version="raw_uncalibrated",
                        resolved_date=datetime.now(timezone.utc)
                    )
                    session.add(po)
                    existing_outcomes[pred.id] = po
                resolved_count += 1

            session.commit()
            logger.info(f"Resolved {resolved_count} PredictionOutcome records.")
            return resolved_count
        finally:
            if should_close:
                session.close()

    def resolve_opportunity_outcomes(self, limit: Optional[int] = None, force_recompute: bool = False) -> int:
        """
        Idempotently resolves OpportunityHistory records into OpportunityOutcome entities.
        """
        session = self._get_session()
        should_close = self._external_session is None
        try:
            if not self._bars_cache:
                self.preload_market_bars()

            existing_outcomes = {
                r.opportunity_id: r for r in session.query(OpportunityOutcome).filter(OpportunityOutcome.opportunity_id.isnot(None)).all()
            }

            opps_query = session.query(OpportunityHistory)
            if limit:
                opps_query = opps_query.limit(limit)

            pending_opps = opps_query.all()
            resolved_count = 0

            for opp in pending_opps:
                existing = existing_outcomes.get(opp.id)
                if existing and not force_recompute and existing.status == "RESOLVED" and existing.return_60d is not None:
                    continue

                disc_date = opp.discovery_date
                outcomes = self.get_forward_outcomes(opp.ticker, disc_date)
                if not outcomes or outcomes["status"] == "UNRESOLVABLE":
                    continue

                r1 = outcomes.get("return_1d")
                r5 = outcomes.get("return_5d")
                r10 = outcomes.get("return_10d")
                r20 = outcomes.get("return_20d")
                r60 = outcomes.get("return_60d")

                success = bool(r20 is not None and r20 > 0.0) if r20 is not None else None
                sector = self._sectors_cache.get(opp.ticker, "Technology")

                if existing:
                    existing.status = outcomes["status"]
                    existing.entry_price = outcomes.get("entry_price")
                    existing.max_drawdown = outcomes.get("max_drawdown_20d")
                    existing.horizon_status = outcomes.get("horizon_status")
                    existing.return_1d = r1
                    existing.return_5d = r5
                    existing.return_10d = r10
                    existing.return_20d = r20
                    existing.return_60d = r60
                    existing.success = success
                    existing.resolved_date = datetime.now(timezone.utc)
                else:
                    oo = OpportunityOutcome(
                        opportunity_id=opp.id,
                        ticker=opp.ticker,
                        discovery_date=datetime.combine(opp.discovery_date, datetime.min.time()),
                        opportunity_score=float(opp.score),
                        status=outcomes["status"],
                        entry_price=outcomes.get("entry_price"),
                        max_drawdown=outcomes.get("max_drawdown_20d"),
                        horizon_status=outcomes.get("horizon_status"),
                        return_1d=r1,
                        return_5d=r5,
                        return_10d=r10,
                        return_20d=r20,
                        return_60d=r60,
                        success=success,
                        market_regime=opp.market_regime or "Bull",
                        sector=sector,
                        resolved_date=datetime.now(timezone.utc)
                    )
                    session.add(oo)
                    existing_outcomes[opp.id] = oo
                resolved_count += 1

            session.commit()
            logger.info(f"Resolved {resolved_count} OpportunityOutcome records.")
            return resolved_count
        finally:
            if should_close:
                session.close()

    def backfill_historical_outcomes(self, include_test_features: bool = True) -> Dict[str, int]:
        """
        Idempotent full historical backfill command.
        Resolves predictions, opportunities, and committee decisions.
        Optionally backfills out-of-sample predictions from TrainingFeature dataset.
        """
        logger.info("Executing idempotent historical outcome backfill...")
        pred_count = self.resolve_prediction_outcomes(force_recompute=True)
        opp_count = self.resolve_opportunity_outcomes(force_recompute=True)
        comm_count = self.resolve_committee_decisions()

        tf_pred_count = 0
        if include_test_features:
            session = self._get_session()
            should_close = self._external_session is None
            try:
                from database.models import TrainingFeature
                import joblib
                from config import ARTIFACTS_DIR
                import json

                model = joblib.load(ARTIFACTS_DIR / "best_model.pkl")
                with open(ARTIFACTS_DIR / "feature_list.json", "r") as f:
                    feature_names = json.load(f)

                existing_po_keys = {
                    r[0] for r in session.query(PredictionOutcome.prediction_id).filter(PredictionOutcome.prediction_id.isnot(None)).all()
                }

                # Query out-of-sample test features
                test_records = (
                    session.query(TrainingFeature)
                    .order_by(TrainingFeature.date.asc())
                    .offset(24020)
                    .limit(6005)
                    .all()
                )

                for tf in test_records:
                    pred_key = f"tf_{tf.ticker}_{tf.date.strftime('%Y%m%d')}"
                    if pred_key in existing_po_keys:
                        continue

                    row_vals = [getattr(tf, col, 0.0) or 0.0 for col in feature_names]
                    x_arr = np.array(row_vals).reshape(1, -1)
                    prob = float(model.predict_proba(x_arr)[0, 1])
                    threshold = 0.60
                    pred_dir = "BUY" if prob >= threshold else "NOT BUY"

                    outcomes = self.get_forward_outcomes(tf.ticker, tf.date)
                    if not outcomes or outcomes["status"] == "UNRESOLVABLE":
                        continue

                    r1 = outcomes.get("return_1d")
                    r5 = outcomes.get("return_5d")
                    r10 = outcomes.get("return_10d")
                    r20 = outcomes.get("return_20d")
                    r60 = outcomes.get("return_60d")

                    c1 = ((r1 > 0) == (pred_dir == "BUY")) if r1 is not None else None
                    c5 = ((r5 > 0) == (pred_dir == "BUY")) if r5 is not None else None
                    c10 = ((r10 > 0) == (pred_dir == "BUY")) if r10 is not None else None
                    c20 = ((r20 > 0) == (pred_dir == "BUY")) if r20 is not None else None
                    c60 = ((r60 > 0) == (pred_dir == "BUY")) if r60 is not None else None

                    y_true = 1.0 if (r5 is not None and r5 > 0) else (0.0 if r5 is not None else None)
                    brier = round(float((prob - y_true) ** 2), 5) if y_true is not None else None
                    sector = self._sectors_cache.get(tf.ticker, "Technology")

                    po = PredictionOutcome(
                        prediction_id=pred_key,
                        ticker=tf.ticker,
                        prediction_date=datetime.combine(tf.date, datetime.min.time()),
                        predicted_probability=prob,
                        threshold=threshold,
                        predicted_direction=pred_dir,
                        status=outcomes["status"],
                        entry_price=outcomes.get("entry_price"),
                        exit_price_5d=outcomes.get("exit_price_5d"),
                        max_drawdown=outcomes.get("max_drawdown_20d"),
                        horizon_status=outcomes.get("horizon_status"),
                        actual_return_1d=r1,
                        actual_return_5d=r5,
                        actual_return_10d=r10,
                        actual_return_20d=r20,
                        actual_return_60d=r60,
                        actual_direction=outcomes.get("actual_direction"),
                        correct_1d=c1,
                        correct_5d=c5,
                        correct_10d=c10,
                        correct_20d=c20,
                        correct_60d=c60,
                        brier_score=brier,
                        market_regime="Bull",
                        sector=sector,
                        model_version="SPRINT_9_BASELINE",
                        feature_version="v2.0_43_features",
                        threshold_version="calibrated_0.60",
                        calibration_version="raw_uncalibrated",
                        resolved_date=datetime.now(timezone.utc)
                    )
                    session.add(po)
                    existing_po_keys.add(pred_key)
                    tf_pred_count += 1

                session.commit()
                logger.info(f"Backfilled {tf_pred_count} out-of-sample PredictionOutcome records from TrainingFeatures.")
            finally:
                if should_close:
                    session.close()

        total_predictions = pred_count + tf_pred_count
        return {
            "prediction_outcomes": total_predictions,
            "opportunity_outcomes": opp_count,
            "committee_decisions": comm_count
        }

    def resolve_committee_decisions(self) -> int:
        """
        Resolves realized 1d, 5d, 10d, 20d, 60d returns and max drawdown for CommitteeDecision records.
        """
        session = self._get_session()
        should_close = self._external_session is None
        try:
            if not self._bars_cache:
                self.preload_market_bars()

            decisions = session.query(CommitteeDecision).all()
            resolved_count = 0

            for cd in decisions:
                d_date = cd.decision_date or cd.timestamp.date()
                outcomes = self.get_forward_outcomes(cd.ticker, d_date)
                if not outcomes or outcomes.get("status") == "UNRESOLVABLE" or outcomes.get("available_forward_sessions", 0) < 1:
                    continue

                cd.realized_return = outcomes.get("return_5d")
                cd.realized_return_5d = outcomes.get("return_5d")
                cd.realized_return_10d = outcomes.get("return_10d")
                cd.realized_return_20d = outcomes.get("return_20d")
                cd.max_drawdown_5d = outcomes.get("max_drawdown_20d")
                cd.outcome_date = outcomes["entry_date"] + timedelta(days=7) if outcomes.get("entry_date") else None

                is_bullish = cd.decision in ["STRONG BUY", "BUY"]
                if cd.realized_return_5d is not None:
                    cd.correct_direction = bool((cd.realized_return_5d > 0) == is_bullish)
                resolved_count += 1

            session.commit()
            logger.info(f"Resolved outcomes for {resolved_count} CommitteeDecision records.")
            return resolved_count
        finally:
            if should_close:
                session.close()

    def inspect_sample_outcomes(self, tickers: List[str] = ["NVDA", "AAPL", "JPM", "XOM"],
                                sample_date: date = date(2024, 1, 10)) -> Dict[str, Any]:
        """
        Phase 2 Gate: Manually inspect sample outcomes for specified tickers on an unambiguous historical date.
        """
        results = {}
        for ticker in tickers:
            res = self.get_forward_outcomes(ticker, sample_date)
            results[ticker] = res
        return results
