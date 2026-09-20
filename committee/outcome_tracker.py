"""
Outcome Tracker for Sprint 7 — Committee Learning & Performance Attribution.

Fetches realized market returns from Yahoo Finance and resolves pending
committee decisions by computing:
  - 5-day, 10-day, and 20-day forward returns
  - Max intra-period drawdown (5d window)
  - Post-decision realized volatility (5d window)

Links outcomes back to CommitteeDecision records to enable:
  - Agent accuracy evaluation
  - Evidence stream attribution
  - Regime-stratified performance analysis
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional

import numpy as np
import yfinance as yf

from committee.committee_memory import CommitteeMemory

logger = logging.getLogger("OutcomeTracker")


class OutcomeTracker:
    """Resolves committee decisions against realized market outcomes."""

    def __init__(self):
        self.memory = CommitteeMemory()

    def resolve_all_pending(self, min_days: int = 7) -> Dict[str, Any]:
        """
        Resolves all pending committee decisions that are old enough
        for 5-day returns to have materialized.

        Args:
            min_days: Minimum calendar days since decision before attempting
                      resolution (default 7 to cover weekends/holidays).

        Returns:
            Summary dict with resolved_count, skipped_count, failed_count, details.
        """
        pending = self.memory.get_pending_decisions()
        cutoff = datetime.now(timezone.utc) - timedelta(days=min_days)

        resolved = 0
        skipped = 0
        failed = 0
        details = []

        for dec in pending:
            dec_time = dec.get("timestamp")
            if dec_time:
                dec_dt = datetime.fromisoformat(dec_time)
                if dec_dt.tzinfo is None:
                    dec_dt = dec_dt.replace(tzinfo=timezone.utc)
                if dec_dt > cutoff:
                    skipped += 1
                    details.append({
                        "id": dec["id"],
                        "ticker": dec["ticker"],
                        "status": "SKIPPED",
                        "reason": f"Only {(datetime.now(timezone.utc) - dec_dt).days}d old, need {min_days}d"
                    })
                    continue

            result = self.resolve_single(dec["id"], dec["ticker"], dec.get("decision_date"))
            if result["status"] == "RESOLVED":
                resolved += 1
            else:
                failed += 1
            details.append(result)

        summary = {
            "total_pending": len(pending),
            "resolved": resolved,
            "skipped": skipped,
            "failed": failed,
            "details": details
        }
        logger.info(
            f"Outcome Resolution: {resolved} resolved, {skipped} skipped (too recent), {failed} failed "
            f"out of {len(pending)} pending"
        )
        return summary

    def resolve_single(
        self,
        decision_id: str,
        ticker: str,
        decision_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Resolves a single committee decision against realized market data.

        Fetches close prices from yfinance, computes:
          - 5d return, 10d return, 20d return
          - Max drawdown within 5d window
          - Post-decision realized volatility (annualized from 5d daily returns)
        """
        try:
            # Parse decision date
            if decision_date:
                if isinstance(decision_date, str):
                    d_date = datetime.fromisoformat(decision_date).date()
                else:
                    d_date = decision_date
            else:
                d_date = datetime.now(timezone.utc).date() - timedelta(days=10)

            # Fetch price data: decision_date - 1d to decision_date + 25d
            start = d_date - timedelta(days=2)
            end = d_date + timedelta(days=30)

            data = yf.download(ticker, start=start.isoformat(), end=end.isoformat(), progress=False)
            if data.empty or len(data) < 2:
                logger.warning(f"Insufficient price data for {ticker} around {d_date}")
                return {
                    "id": decision_id,
                    "ticker": ticker,
                    "status": "FAILED",
                    "reason": "Insufficient price data from yfinance"
                }

            # Handle multi-level columns from yfinance
            if hasattr(data.columns, 'levels') and len(data.columns.levels) > 1:
                data.columns = data.columns.get_level_values(0)

            closes = data["Close"].dropna()

            # Find the first trading day on or after decision_date
            trading_dates = closes.index
            base_candidates = trading_dates[trading_dates >= str(d_date)]
            if len(base_candidates) == 0:
                return {
                    "id": decision_id,
                    "ticker": ticker,
                    "status": "FAILED",
                    "reason": f"No trading days found on or after {d_date}"
                }

            base_idx = trading_dates.get_loc(base_candidates[0])
            base_price = float(closes.iloc[base_idx])

            # Compute returns for each horizon
            returns = {}
            horizons = {"5d": 5, "10d": 10, "20d": 20}
            for label, offset in horizons.items():
                target_idx = base_idx + offset
                if target_idx < len(closes):
                    target_price = float(closes.iloc[target_idx])
                    returns[label] = (target_price - base_price) / base_price
                else:
                    returns[label] = None

            # Max drawdown within 5d window
            max_dd = 0.0
            dd_end = min(base_idx + 6, len(closes))
            window_prices = closes.iloc[base_idx:dd_end].values.astype(float)
            if len(window_prices) > 1:
                peak = window_prices[0]
                for p in window_prices[1:]:
                    if p > peak:
                        peak = p
                    dd = (p - peak) / peak
                    if dd < max_dd:
                        max_dd = dd

            # Post-decision realized volatility (annualized)
            post_vol = None
            if len(window_prices) > 2:
                daily_returns = np.diff(window_prices) / window_prices[:-1]
                post_vol = float(np.std(daily_returns) * np.sqrt(252))

            # Resolve if we have at least 5d return
            if returns.get("5d") is None:
                return {
                    "id": decision_id,
                    "ticker": ticker,
                    "status": "FAILED",
                    "reason": "5-day return not yet available"
                }

            success = self.memory.update_outcomes(
                decision_id=decision_id,
                realized_return_5d=returns["5d"],
                realized_return_10d=returns.get("10d"),
                realized_return_20d=returns.get("20d"),
                max_drawdown_5d=max_dd,
                post_decision_volatility=post_vol
            )

            if success:
                logger.info(
                    f"Resolved {ticker} Decision #{decision_id[:8]}...: "
                    f"5d={returns['5d']:+.2%}, 10d={returns.get('10d', 'N/A')}, "
                    f"20d={returns.get('20d', 'N/A')}, MaxDD={max_dd:.2%}, Vol={post_vol or 0:.2%}"
                )
                return {
                    "id": decision_id,
                    "ticker": ticker,
                    "status": "RESOLVED",
                    "returns": returns,
                    "max_drawdown_5d": max_dd,
                    "post_decision_volatility": post_vol
                }
            else:
                return {
                    "id": decision_id,
                    "ticker": ticker,
                    "status": "FAILED",
                    "reason": "CommitteeMemory.update_outcomes() returned False"
                }

        except Exception as e:
            logger.error(f"Failed to resolve decision {decision_id} for {ticker}: {e}")
            return {
                "id": decision_id,
                "ticker": ticker,
                "status": "FAILED",
                "reason": str(e)
            }

    def get_resolution_summary(self) -> Dict[str, Any]:
        """Returns high-level counts of resolved vs pending vs total decisions."""
        resolved = self.memory.get_all_resolved_decisions()
        pending = self.memory.get_pending_decisions()
        return {
            "total": len(resolved) + len(pending),
            "resolved": len(resolved),
            "pending": len(pending),
            "resolution_rate": len(resolved) / max(len(resolved) + len(pending), 1)
        }


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    tracker = OutcomeTracker()

    print("--- 1. Resolution Summary ---")
    summary = tracker.get_resolution_summary()
    print(json.dumps(summary, indent=2))

    print("\n--- 2. Resolving All Pending Decisions ---")
    result = tracker.resolve_all_pending()
    print(f"Resolved: {result['resolved']}, Skipped: {result['skipped']}, Failed: {result['failed']}")
    for d in result["details"]:
        print(f"  {d['ticker']} [{d['status']}] {d.get('reason', d.get('returns', ''))}")
