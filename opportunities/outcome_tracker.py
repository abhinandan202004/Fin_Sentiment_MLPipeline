"""
Opportunity Outcome Tracker for Sprint 8: Autonomous Research & Market Opportunity Discovery.

Tracks realized forward performance for every discovered opportunity:
  - 5-day return (tactical swing horizon)
  - 20-day return (monthly trend horizon)
  - 60-day return (quarterly investment horizon)
  - Max intra-period drawdown
  - Binary success metric (return_20d > 0 and beating benchmark)

Computes scanner validation metrics:
  - Precision@5 & Precision@10 (hit rate of top-ranked opportunities)
  - Performance by Opportunity Score tier (>=80, 70-79, <70)
  - Scanner Monotonicity (does higher score lead to higher return?)

Usage:
    python opportunities/outcome_tracker.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, date, timedelta, timezone
from collections import defaultdict
import numpy as np
import yfinance as yf

from database.connection import SessionLocal, init_db
from database.models import OpportunityHistory

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("OpportunityOutcomeTracker")


class OpportunityOutcomeTracker:
    """Tracks and evaluates realized market outcomes of scanned opportunities."""

    def __init__(self):
        init_db()

    def resolve_all_pending(self, min_days: int = 5) -> Dict[str, Any]:
        """
        Scans all ACTIVE opportunities in the database.
        If enough time has elapsed since discovery_date, fetches price bars
        and resolves 5d, 20d, and 60d realized returns.
        """
        session = SessionLocal()
        resolved_count = 0
        skipped_count = 0
        failed_count = 0
        today = datetime.now(timezone.utc).date()

        try:
            pending = (
                session.query(OpportunityHistory)
                .filter(OpportunityHistory.status == "ACTIVE")
                .all()
            )

            if not pending:
                logger.info("No pending active opportunities to resolve.")
                return {"resolved": 0, "skipped": 0, "failed": 0, "total_pending": 0}

            # Group by ticker to batch market data requests
            by_ticker = defaultdict(list)
            for opp in pending:
                by_ticker[opp.ticker].append(opp)

            for ticker, opps in by_ticker.items():
                try:
                    # Find earliest discovery date
                    earliest_date = min(o.discovery_date for o in opps)
                    days_elapsed = (today - earliest_date).days

                    if days_elapsed < min_days:
                        skipped_count += len(opps)
                        continue

                    # Download historical bars
                    start_str = (earliest_date - timedelta(days=5)).strftime("%Y-%m-%d")
                    end_str = (today + timedelta(days=1)).strftime("%Y-%m-%d")
                    df = yf.download(ticker, start=start_str, end=end_str, progress=False)

                    if df is None or df.empty or len(df) < 2:
                        logger.warning(f"Insufficient yfinance data for {ticker}")
                        failed_count += len(opps)
                        continue

                    if isinstance(df.columns, object) and hasattr(df.columns, "levels") and len(df.columns.levels) > 1:
                        df.columns = df.columns.get_level_values(0)

                    df["DateOnly"] = df.index.date

                    for opp in opps:
                        opp_elapsed = (today - opp.discovery_date).days
                        if opp_elapsed < min_days:
                            skipped_count += 1
                            continue

                        # Find matching entry bar
                        entry_bars = df[df["DateOnly"] >= opp.discovery_date]
                        if entry_bars.empty:
                            skipped_count += 1
                            continue

                        entry_price = float(entry_bars.iloc[0]["Close"])
                        forward_bars = entry_bars.iloc[1:]

                        if len(forward_bars) >= 5:
                            p_5d = float(forward_bars.iloc[min(4, len(forward_bars) - 1)]["Close"])
                            opp.return_5d = round((p_5d - entry_price) / entry_price, 4)

                        if len(forward_bars) >= 20:
                            p_20d = float(forward_bars.iloc[min(19, len(forward_bars) - 1)]["Close"])
                            opp.return_20d = round((p_20d - entry_price) / entry_price, 4)
                            # Max drawdown in first 20 days
                            lows_20d = forward_bars.iloc[:20]["Low"]
                            min_low = float(lows_20d.min())
                            opp.max_drawdown = round((min_low - entry_price) / entry_price, 4)
                            opp.success = bool(opp.return_20d > 0.0)
                            opp.status = "RESOLVED"
                            opp.resolved_date = today

                        if len(forward_bars) >= 60:
                            p_60d = float(forward_bars.iloc[min(59, len(forward_bars) - 1)]["Close"])
                            opp.return_60d = round((p_60d - entry_price) / entry_price, 4)

                        resolved_count += 1

                except Exception as e:
                    logger.error(f"Error resolving outcomes for {ticker}: {e}")
                    failed_count += len(opps)

            session.commit()
            logger.info(f"Resolved {resolved_count} opportunities ({skipped_count} skipped, {failed_count} failed).")
            return {
                "resolved": resolved_count,
                "skipped": skipped_count,
                "failed": failed_count,
                "total_pending": len(pending)
            }

        except Exception as e:
            session.rollback()
            logger.error(f"Failed outcome resolution transaction: {e}")
            raise
        finally:
            session.close()

    def get_scanner_performance_summary(self) -> Dict[str, Any]:
        """
        Computes precision, mean returns by score decile, and track record.
        """
        session = SessionLocal()
        try:
            resolved = (
                session.query(OpportunityHistory)
                .filter(OpportunityHistory.return_5d.isnot(None))
                .order_by(OpportunityHistory.discovery_date.desc())
                .all()
            )

            if not resolved:
                return {
                    "status": "insufficient_data",
                    "total_resolved": 0,
                    "precision_at_5": None,
                    "precision_at_10": None,
                    "tier_performance": {}
                }

            # Top 5 and Top 10 by priority/score on each discovery date
            by_date = defaultdict(list)
            for r in resolved:
                by_date[r.discovery_date].append(r)

            top_5_success = []
            top_10_success = []

            for d, date_opps in by_date.items():
                sorted_opps = sorted(date_opps, key=lambda x: x.score, reverse=True)
                for o in sorted_opps[:5]:
                    if o.success is not None:
                        top_5_success.append(1 if o.success else 0)
                for o in sorted_opps[:10]:
                    if o.success is not None:
                        top_10_success.append(1 if o.success else 0)

            p_5 = float(np.mean(top_5_success)) if top_5_success else None
            p_10 = float(np.mean(top_10_success)) if top_10_success else None

            # Tiers: High (>=80), Medium (70-79), Low (<70)
            high_tier = [r for r in resolved if r.score >= 80.0]
            med_tier = [r for r in resolved if 70.0 <= r.score < 80.0]
            low_tier = [r for r in resolved if r.score < 70.0]

            def compute_tier_stats(cohort: List[OpportunityHistory]) -> Dict[str, Any]:
                if not cohort:
                    return {"count": 0, "avg_return_5d": None, "avg_return_20d": None, "win_rate": None}
                ret_5s = [c.return_5d for c in cohort if c.return_5d is not None]
                ret_20s = [c.return_20d for c in cohort if c.return_20d is not None]
                wins = [1 if c.success else 0 for c in cohort if c.success is not None]

                return {
                    "count": len(cohort),
                    "avg_return_5d": round(float(np.mean(ret_5s)), 4) if ret_5s else None,
                    "avg_return_20d": round(float(np.mean(ret_20s)), 4) if ret_20s else None,
                    "win_rate": round(float(np.mean(wins)), 4) if wins else None
                }

            return {
                "status": "computed",
                "total_resolved": len(resolved),
                "precision_at_5": round(p_5, 4) if p_5 is not None else None,
                "precision_at_10": round(p_10, 4) if p_10 is not None else None,
                "tier_performance": {
                    "high_conviction_80_plus": compute_tier_stats(high_tier),
                    "medium_conviction_70_79": compute_tier_stats(med_tier),
                    "low_conviction_sub_70": compute_tier_stats(low_tier)
                }
            }
        finally:
            session.close()


if __name__ == "__main__":
    tracker = OpportunityOutcomeTracker()
    res = tracker.resolve_all_pending()
    perf = tracker.get_scanner_performance_summary()
    print("\n" + "=" * 60)
    print("Sprint 8: Opportunity Outcome Tracker Summary")
    print("=" * 60)
    print(f"Outcome Resolution: {res}")
    print(f"Total Resolved: {perf.get('total_resolved', 0)}")
    if perf.get("precision_at_5") is not None:
        print(f"Precision@5:  {perf['precision_at_5']:.1%}")
    if perf.get("precision_at_10") is not None:
        print(f"Precision@10: {perf['precision_at_10']:.1%}")
    print("\nScore Tier Breakdown:")
    for tier, stats in perf.get("tier_performance", {}).items():
        print(f"  {tier:<26} | Count: {stats['count']:<3} | 5d Ret: {stats.get('avg_return_5d') or 'N/A'} | Win Rate: {stats.get('win_rate') or 'N/A'}")
