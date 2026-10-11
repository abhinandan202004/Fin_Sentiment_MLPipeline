"""
Historical Outcome Backfill Script for Fin_Sentiment_MLPipeline.
Sprint 9.2: Autonomous Evaluation & Outcome Resolution.

Executes idempotent backfill for:
  - Prediction outcomes across 1d, 5d, 10d, 20d, 60d horizons
  - Opportunity outcomes with success tracking
  - Committee decisions forward returns and drawdowns
"""

import sys
import logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database.connection import init_db
from evaluation.outcome_tracker import OutcomeTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("BackfillOutcomes")


def main():
    logger.info("Initializing database schema...")
    init_db()

    logger.info("Initializing OutcomeTracker and preloading market bars...")
    tracker = OutcomeTracker()
    tracker.preload_market_bars()

    logger.info("Starting idempotent historical outcome backfill...")
    results = tracker.backfill_historical_outcomes(include_test_features=True)

    print("\n========================================================")
    print(" OUTCOME BACKFILL COMPLETE (IDEMPOTENT & MULTI-HORIZON)")
    print("========================================================")
    print(f"  Prediction Outcomes Resolved:  {results['prediction_outcomes']:,}")
    print(f"  Opportunity Outcomes Resolved: {results['opportunity_outcomes']:,}")
    print(f"  Committee Decisions Resolved:  {results['committee_decisions']:,}")
    print("========================================================\n")


if __name__ == "__main__":
    main()
