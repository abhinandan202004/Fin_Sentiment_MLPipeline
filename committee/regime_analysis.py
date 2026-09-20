"""
Regime Analysis for Sprint 7 — Committee Learning & Performance Attribution.

Stratifies all performance metrics by detected market regime:
  - Per-regime: accuracy, avg returns (5d/10d/20d), drawdown stats
  - Agent × Regime cross-tabulation matrix
  - Regime-specific confidence recommendations
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
from typing import Dict, List, Any, Optional
from collections import defaultdict

import numpy as np

from committee.committee_memory import CommitteeMemory

logger = logging.getLogger("RegimeAnalysis")

REGIMES = ["Bull", "Bear", "Sideways", "High Volatility", "Low Volatility"]


class RegimeAnalysis:
    """Stratifies committee performance by market regime."""

    def __init__(self):
        self.memory = CommitteeMemory()

    def compute_regime_performance(self) -> Dict[str, Any]:
        """
        Groups all resolved decisions by market_regime and computes:
          - total_decisions, correct_decisions, accuracy
          - avg_return_5d, avg_return_10d, avg_return_20d
          - avg_drawdown, max_drawdown (worst single decision)
          - avg_post_decision_volatility
          - best_decision (highest return), worst_decision (worst return)
        """
        all_decisions = self.memory.get_all_resolved_decisions()
        if not all_decisions:
            return {"status": "no_data", "regimes": {}}

        by_regime = defaultdict(list)
        for dec in all_decisions:
            regime = dec.get("market_regime") or "Unknown"
            by_regime[regime].append(dec)

        results = {}
        for regime, decisions in by_regime.items():
            returns_5d = [d["realized_return_5d"] for d in decisions if d.get("realized_return_5d") is not None]
            returns_10d = [d["realized_return_10d"] for d in decisions if d.get("realized_return_10d") is not None]
            returns_20d = [d["realized_return_20d"] for d in decisions if d.get("realized_return_20d") is not None]
            drawdowns = [d["max_drawdown_5d"] for d in decisions if d.get("max_drawdown_5d") is not None]
            volatilities = [d["post_decision_volatility"] for d in decisions if d.get("post_decision_volatility") is not None]
            correct = [d for d in decisions if d.get("correct_direction") is True]

            best = max(decisions, key=lambda d: d.get("realized_return_5d", -999))
            worst = min(decisions, key=lambda d: d.get("realized_return_5d", 999))

            results[regime] = {
                "total_decisions": len(decisions),
                "correct_decisions": len(correct),
                "accuracy": round(len(correct) / len(decisions), 4) if decisions else 0.0,
                "avg_return_5d": round(float(np.mean(returns_5d)), 4) if returns_5d else None,
                "avg_return_10d": round(float(np.mean(returns_10d)), 4) if returns_10d else None,
                "avg_return_20d": round(float(np.mean(returns_20d)), 4) if returns_20d else None,
                "avg_drawdown": round(float(np.mean(drawdowns)), 4) if drawdowns else None,
                "max_drawdown": round(float(min(drawdowns)), 4) if drawdowns else None,
                "avg_post_decision_volatility": round(float(np.mean(volatilities)), 4) if volatilities else None,
                "best_decision": {
                    "ticker": best["ticker"],
                    "decision": best["decision"],
                    "return_5d": best.get("realized_return_5d")
                },
                "worst_decision": {
                    "ticker": worst["ticker"],
                    "decision": worst["decision"],
                    "return_5d": worst.get("realized_return_5d")
                }
            }

        return {
            "status": "computed",
            "total_resolved": len(all_decisions),
            "regimes": results
        }

    def compute_regime_agent_matrix(self) -> Dict[str, Any]:
        """
        Cross-tabulates agent accuracy × market regime.

        Returns a matrix structure:
        {
            "Bull": {"BullAnalyst": 0.72, "BearAnalyst": 0.45, ...},
            "Bear": {"BullAnalyst": 0.38, "BearAnalyst": 0.71, ...},
            ...
        }
        """
        votes = self.memory.get_all_votes_with_outcomes()
        if not votes:
            return {"status": "no_data", "matrix": {}}

        # Group by regime × agent
        regime_agent = defaultdict(lambda: defaultdict(list))
        for v in votes:
            regime = v.get("market_regime") or "Unknown"
            agent = v["agent_name"]
            regime_agent[regime][agent].append(1 if v["is_accurate"] else 0)

        matrix = {}
        for regime, agents in regime_agent.items():
            matrix[regime] = {}
            for agent, accuracies in agents.items():
                matrix[regime][agent] = {
                    "accuracy": round(sum(accuracies) / len(accuracies), 4),
                    "sample_size": len(accuracies)
                }

        return {
            "status": "computed",
            "total_votes": len(votes),
            "matrix": matrix
        }

    def get_regime_recommendation(self, regime: str) -> Dict[str, Any]:
        """
        Based on historical performance, recommends whether to increase/decrease
        committee confidence for a given regime.

        Returns confidence_multiplier:
          > 1.0 = regime has historically outperformed → boost confidence
          < 1.0 = regime has historically underperformed → dampen confidence
          = 1.0 = neutral
        """
        regime_perf = self.compute_regime_performance()
        if regime_perf["status"] == "no_data":
            return {"regime": regime, "recommendation": "neutral", "multiplier": 1.0, "reason": "No data"}

        regime_data = regime_perf["regimes"].get(regime)
        if not regime_data or regime_data["total_decisions"] < 3:
            return {
                "regime": regime,
                "recommendation": "neutral",
                "multiplier": 1.0,
                "reason": f"Insufficient samples ({regime_data['total_decisions'] if regime_data else 0})"
            }

        accuracy = regime_data["accuracy"]
        avg_return = regime_data.get("avg_return_5d", 0) or 0
        avg_dd = regime_data.get("avg_drawdown", 0) or 0

        # Scoring: accuracy > 60% and positive avg return → boost
        if accuracy >= 0.65 and avg_return > 0.005:
            multiplier = min(1.0 + (accuracy - 0.50) * 0.5, 1.25)
            recommendation = "increase_confidence"
            reason = f"Accuracy {accuracy:.0%} with avg +{avg_return:.2%} return in {regime} regime"
        elif accuracy < 0.45 or avg_return < -0.005:
            multiplier = max(1.0 - (0.50 - accuracy) * 0.5, 0.75)
            recommendation = "decrease_confidence"
            reason = f"Below-average accuracy ({accuracy:.0%}) or negative avg return ({avg_return:+.2%}) in {regime} regime"
        else:
            multiplier = 1.0
            recommendation = "neutral"
            reason = f"Balanced performance ({accuracy:.0%} accuracy, {avg_return:+.2%} avg return) in {regime} regime"

        return {
            "regime": regime,
            "recommendation": recommendation,
            "multiplier": round(multiplier, 3),
            "accuracy": accuracy,
            "avg_return_5d": avg_return,
            "avg_drawdown": avg_dd,
            "sample_size": regime_data["total_decisions"],
            "reason": reason
        }


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    ra = RegimeAnalysis()

    print("--- 1. Regime Performance ---")
    perf = ra.compute_regime_performance()
    print(json.dumps(perf, indent=2, default=str))

    print("\n--- 2. Regime × Agent Matrix ---")
    matrix = ra.compute_regime_agent_matrix()
    print(json.dumps(matrix, indent=2, default=str))

    print("\n--- 3. Regime Recommendations ---")
    for regime in REGIMES:
        rec = ra.get_regime_recommendation(regime)
        print(f"  {regime}: {rec['recommendation']} (×{rec['multiplier']}) — {rec['reason']}")
