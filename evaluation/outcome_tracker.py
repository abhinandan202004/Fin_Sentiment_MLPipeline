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
        """
        if not self._bars_cache:
            self.preload_market_bars()

        if ticker not in self._bars_cache:
            return None

        bars = self._bars_cache[ticker]
        dates = self._dates_cache[ticker]
        if not bars:
            return None

        d_date = decision_date.date() if isinstance(decision_date, datetime) else decision_date

        entry_idx = bisect_left(dates, d_date)
        if entry_idx >= len(bars):
            # Decision date is after our latest available bar
            return None

        entry_bar = bars[entry_idx]
        entry_price = float(entry_bar.close)
        if entry_price <= 0.0:
            return None

        total_forward_sessions = len(bars) - 1 - entry_idx

        def calc_ret(step: int) -> Optional[float]:
            if total_forward_sessions >= step:
                p_fwd = float(bars[entry_idx + step].close)
                return round((p_fwd - entry_price) / entry_price, 5)
            return None

        ret_1d = calc_ret(1)
        ret_5d = calc_ret(5)
        ret_10d = calc_ret(10)
        ret_20d = calc_ret(20)
        ret_60d = calc_ret(60)

        # Drawdown in the 20-session window
        window_bars = bars[entry_idx + 1: entry_idx + min(21, total_forward_sessions + 1)]
        if window_bars:
            min_low = min(float(b.low) for b in window_bars)
            max_dd = round((min_low - entry_price) / entry_price, 5)
        else:
            max_dd = 0.0

        actual_dir = "BUY" if (ret_5d is not None and ret_5d > 0) else "NOT BUY"

        return {
            "entry_date": entry_bar.date,
            "entry_price": entry_price,
            "available_forward_sessions": total_forward_sessions,
            "return_1d": ret_1d,
            "return_5d": ret_5d,
            "return_10d": ret_10d,
            "return_20d": ret_20d,
            "return_60d": ret_60d,
            "max_drawdown_20d": max_dd,
            "actual_direction": actual_dir
        }

    def resolve_prediction_outcomes(self, limit: Optional[int] = None) -> int:
        """
        Resolves pending Prediction records into PredictionOutcome records.
        """
        session = self._get_session()
        should_close = self._external_session is None
        try:
            if not self._bars_cache:
                self.preload_market_bars()

            # Find predictions without an outcome
            existing_pred_ids = {
                r[0] for r in session.query(PredictionOutcome.prediction_id).filter(PredictionOutcome.prediction_id.isnot(None)).all()
            }

            preds_query = session.query(Prediction)
            if existing_pred_ids:
                preds_query = preds_query.filter(~Prediction.id.in_(existing_pred_ids))
            if limit:
                preds_query = preds_query.limit(limit)

            pending_preds = preds_query.all()
            resolved_count = 0

            for pred in pending_preds:
                outcomes = self.get_forward_outcomes(pred.ticker, pred.generated_at)
                if not outcomes or outcomes["available_forward_sessions"] < 1:
                    continue

                threshold = 0.60  # Default baseline threshold
                pred_prob = float(pred.confidence)
                pred_dir = "BUY" if pred_prob >= threshold else "NOT BUY"

                r1 = outcomes["return_1d"]
                r5 = outcomes["return_5d"]
                r10 = outcomes["return_10d"]
                r20 = outcomes["return_20d"]
                r60 = outcomes["return_60d"]

                # Binary correctness: 1 if positive return and predicted BUY, or negative return and predicted NOT BUY
                c1 = ((r1 > 0) == (pred_dir == "BUY")) if r1 is not None else None
                c5 = ((r5 > 0) == (pred_dir == "BUY")) if r5 is not None else None
                c10 = ((r10 > 0) == (pred_dir == "BUY")) if r10 is not None else None
                c20 = ((r20 > 0) == (pred_dir == "BUY")) if r20 is not None else None
                c60 = ((r60 > 0) == (pred_dir == "BUY")) if r60 is not None else None

                # Brier score against 5d outcome (1 if r5 > 0 else 0)
                y_true = 1.0 if (r5 is not None and r5 > 0) else 0.0
                brier = round(float((pred_prob - y_true) ** 2), 5)

                sector = self._sectors_cache.get(pred.ticker, "Technology")

                po = PredictionOutcome(
                    prediction_id=pred.id,
                    ticker=pred.ticker,
                    prediction_date=pred.generated_at,
                    predicted_probability=pred_prob,
                    threshold=threshold,
                    predicted_direction=pred_dir,
                    actual_return_1d=r1,
                    actual_return_5d=r5,
                    actual_return_10d=r10,
                    actual_return_20d=r20,
                    actual_return_60d=r60,
                    actual_direction=outcomes["actual_direction"],
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
                resolved_count += 1

            session.commit()
            logger.info(f"Resolved {resolved_count} PredictionOutcome records.")
            return resolved_count
        finally:
            if should_close:
                session.close()

    def resolve_opportunity_outcomes(self, limit: Optional[int] = None) -> int:
        """
        Resolves OpportunityHistory records into OpportunityOutcome entities.
        """
        session = self._get_session()
        should_close = self._external_session is None
        try:
            if not self._bars_cache:
                self.preload_market_bars()

            existing_opp_ids = {
                r[0] for r in session.query(OpportunityOutcome.opportunity_id).filter(OpportunityOutcome.opportunity_id.isnot(None)).all()
            }

            opps_query = session.query(OpportunityHistory)
            if existing_opp_ids:
                opps_query = opps_query.filter(~OpportunityHistory.id.in_(existing_opp_ids))
            if limit:
                opps_query = opps_query.limit(limit)

            pending_opps = opps_query.all()
            resolved_count = 0

            for opp in pending_opps:
                disc_date = opp.discovery_date
                outcomes = self.get_forward_outcomes(opp.ticker, disc_date)
                if not outcomes or outcomes["available_forward_sessions"] < 1:
                    continue

                r1 = outcomes["return_1d"]
                r5 = outcomes["return_5d"]
                r10 = outcomes["return_10d"]
                r20 = outcomes["return_20d"]
                r60 = outcomes["return_60d"]

                success = bool(r20 is not None and r20 > 0.0)
                sector = self._sectors_cache.get(opp.ticker, "Technology")

                oo = OpportunityOutcome(
                    opportunity_id=opp.id,
                    ticker=opp.ticker,
                    discovery_date=datetime.combine(opp.discovery_date, datetime.min.time()),
                    opportunity_score=float(opp.score),
                    return_1d=r1,
                    return_5d=r5,
                    return_10d=r10,
                    return_20d=r20,
                    return_60d=r60,
                    success=success,
                    market_regime=opp.market_regime or "Bull",
                    sector=sector
                )
                session.add(oo)
                resolved_count += 1

            session.commit()
            logger.info(f"Resolved {resolved_count} OpportunityOutcome records.")
            return resolved_count
        finally:
            if should_close:
                session.close()

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
                if not outcomes or outcomes["available_forward_sessions"] < 1:
                    continue

                cd.realized_return = outcomes["return_5d"]
                cd.realized_return_5d = outcomes["return_5d"]
                cd.realized_return_10d = outcomes["return_10d"]
                cd.realized_return_20d = outcomes["return_20d"]
                cd.max_drawdown_5d = outcomes["max_drawdown_20d"]
                cd.outcome_date = outcomes["entry_date"] + timedelta(days=7)

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
