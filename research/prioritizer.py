"""
Research Prioritizer for Sprint 8: Autonomous Research & Market Opportunity Discovery.

Transforms broad market scanner outputs into an actionable, prioritized research queue.

Fuses:
  - Base Opportunity Score (from OpportunityScanner)
  - Catalyst Boost (up to +20 points for positive corporate triggers)
  - Trend Alignment Boost (up to +15 points for leading secular macro themes)
  - Risk Penalty (for negative antitrust or earnings downgrade catalysts)

Outputs:
  A ranked research queue (Priority 1, 2, 3...) ready for analyst or committee deep-dive.

Usage:
    python research/prioritizer.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone

from opportunities.scanner import OpportunityScanner
from agents.trend_agent import TrendAgent
from agents.catalyst_agent import CatalystAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ResearchPrioritizer")


class ResearchPrioritizer:
    """Fuses multi-factor opportunities, catalysts, and thematic trends into a priority queue."""

    def __init__(self):
        self.scanner = OpportunityScanner()
        self.trend_agent = TrendAgent()
        self.catalyst_agent = CatalystAgent()

    def generate_prioritized_queue(self, top_n: int = 10) -> Dict[str, Any]:
        """
        Executes scanner, detects catalysts and trends, applies boosts/penalties,
        and generates a prioritized research queue.
        """
        logger.info("Executing Autonomous Opportunity Scan...")
        scan_results = self.scanner.scan_universe(save_to_db=True)
        opportunities = scan_results.get("opportunities", [])
        regime = scan_results.get("market_regime", "Neutral")

        logger.info("Analyzing secular macro trends...")
        trends = self.trend_agent.snapshot_trends(save_to_db=True)
        trend_growth_map = {t["theme"]: t["mention_growth"] for t in trends}

        logger.info("Scanning for active catalysts...")
        all_catalysts = self.catalyst_agent.scan_all_catalysts(days_back=14, save_to_db=True)

        # Group catalysts by ticker
        ticker_catalysts = {}
        for c in all_catalysts:
            t = c["ticker"]
            if t not in ticker_catalysts or abs(c["expected_impact"]) > abs(ticker_catalysts[t]["expected_impact"]):
                ticker_catalysts[t] = c

        queue = []
        for opp in opportunities:
            ticker = opp["ticker"]
            base_score = opp["opportunity_score"]

            # 1. Catalyst Boost or Penalty (up to +/- 20 points)
            catalyst_info = ticker_catalysts.get(ticker)
            catalyst_boost = 0.0
            catalyst_desc = "No major catalyst detected"
            if catalyst_info:
                # expected_impact is in [-1.0, 1.0]
                catalyst_boost = catalyst_info["expected_impact"] * 20.0
                catalyst_desc = f"{catalyst_info['catalyst_type']}: {catalyst_info['description'][:45]}"

            # 2. Trend Alignment Boost (up to +15 points)
            trend_theme = self.trend_agent.get_theme_for_ticker(ticker)
            trend_boost = 0.0
            if trend_theme and trend_theme in trend_growth_map:
                growth = trend_growth_map[trend_theme]
                # Positive growth gives up to +15 boost
                if growth > 0.5:
                    trend_boost = 15.0
                elif growth > 0.0:
                    trend_boost = 7.5
                else:
                    trend_boost = 0.0

            # Composite Priority Score
            composite_score = base_score + catalyst_boost + trend_boost
            composite_score = max(0.0, min(100.0, round(composite_score, 1)))

            # Recommended Action
            if composite_score >= 80.0:
                action = "IMMEDIATE_COMMITTEE_DEBATE"
            elif composite_score >= 65.0:
                action = "IN_DEPTH_RESEARCH"
            elif composite_score >= 50.0:
                action = "MONITOR_WATCHLIST"
            else:
                action = "DE-PRIORITIZE"

            queue.append({
                "ticker": ticker,
                "composite_score": composite_score,
                "base_opportunity_score": base_score,
                "catalyst_boost": round(catalyst_boost, 1),
                "trend_boost": round(trend_boost, 1),
                "confidence": opp["confidence"],
                "trend_theme": trend_theme or "Diversified",
                "top_catalyst": catalyst_desc,
                "ml_probability": opp.get("ml_probability", 0.50),
                "analog_win_rate": opp.get("analog_win_rate", 0.50),
                "action": action
            })

        # Sort descending by composite_score
        queue.sort(key=lambda x: x["composite_score"], reverse=True)

        for i, item in enumerate(queue):
            item["priority"] = i + 1

        return {
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "market_regime": regime,
            "total_analyzed": len(queue),
            "top_priority": queue[:top_n],
            "full_queue": queue
        }


if __name__ == "__main__":
    prioritizer = ResearchPrioritizer()
    res = prioritizer.generate_prioritized_queue(top_n=8)
    print("\n" + "=" * 80)
    print("Sprint 8: Autonomous Research Prioritizer Queue")
    print(f"Date: {res['date']} | Regime: {res['market_regime']} | Tracked Assets: {res['total_analyzed']}")
    print("=" * 80)
    print(f"{'Priority':<10} {'Ticker':<8} {'Score':<8} {'Base':<8} {'CatBoost':<10} {'Trend':<25} {'Action'}")
    print("-" * 80)
    for q in res["top_priority"]:
        print(f"#{q['priority']:<9} {q['ticker']:<8} {q['composite_score']:<8.1f} {q['base_opportunity_score']:<8.1f} {q['catalyst_boost']:+<10.1f} {q['trend_theme']:<25} {q['action']}")
    print("=" * 80)
