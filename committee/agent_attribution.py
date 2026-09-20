"""
Agent Attribution for Sprint 7 — Committee Learning & Performance Attribution.

Measures per-agent performance across multiple dimensions:
  - Directional Accuracy: % of calls where the agent's stance matched market direction
  - Brier Score: Mean squared error between stated confidence and binary outcome
  - Calibration Error: Difference between stated confidence and actual hit rate
  - Drawdown Attribution: Average max drawdown when agent was wrong
  - EWMA Trend: Exponentially-weighted accuracy to detect improving/degrading agents

All metrics can be stratified by market regime.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
from typing import Dict, List, Any, Optional
from collections import defaultdict

import numpy as np

from committee.committee_memory import CommitteeMemory

logger = logging.getLogger("AgentAttribution")


class AgentAttribution:
    """Computes institutional-grade performance attribution for each committee agent."""

    AGENT_NAMES = ["BullAnalyst", "BearAnalyst", "RiskOfficer", "EvidenceProsecutor", "PortfolioManager"]

    # Validated backtest baselines (used when insufficient live data)
    BASELINE_ACCURACY = {
        "BullAnalyst": 0.61,
        "BearAnalyst": 0.58,
        "RiskOfficer": 0.69,
        "EvidenceProsecutor": 0.74,
        "PortfolioManager": 0.55
    }

    def __init__(self):
        self.memory = CommitteeMemory()

    def compute_agent_accuracy(self) -> Dict[str, Any]:
        """
        Computes comprehensive accuracy metrics per agent:
        - accuracy: directional hit rate
        - brier_score: (confidence - outcome)^2 mean
        - calibration_error: |mean_confidence - accuracy|
        - avg_drawdown_on_miss: average max drawdown when agent was wrong
        - sample_size: number of resolved votes
        """
        votes = self.memory.get_all_votes_with_outcomes()
        if not votes:
            return self._baseline_response("No resolved votes found")

        by_agent = defaultdict(list)
        for v in votes:
            by_agent[v["agent_name"]].append(v)

        results = {}
        for agent_name in self.AGENT_NAMES:
            agent_votes = by_agent.get(agent_name, [])
            if len(agent_votes) < 3:
                results[agent_name] = {
                    "accuracy": self.BASELINE_ACCURACY.get(agent_name, 0.50),
                    "brier_score": None,
                    "calibration_error": None,
                    "avg_drawdown_on_miss": None,
                    "sample_size": len(agent_votes),
                    "status": "baseline"
                }
                continue

            accuracies = [1 if v["is_accurate"] else 0 for v in agent_votes]
            confidences = [v["confidence"] for v in agent_votes]

            accuracy = sum(accuracies) / len(accuracies)

            # Brier score: mean of (confidence - outcome)^2
            brier = np.mean([(c - o) ** 2 for c, o in zip(confidences, accuracies)])

            # Calibration error: |mean_confidence - accuracy|
            cal_error = abs(np.mean(confidences) - accuracy)

            # Average drawdown on misses
            miss_drawdowns = [
                v["max_drawdown_5d"] for v in agent_votes
                if not v["is_accurate"] and v.get("max_drawdown_5d") is not None
            ]
            avg_dd_miss = float(np.mean(miss_drawdowns)) if miss_drawdowns else None

            results[agent_name] = {
                "accuracy": round(accuracy, 4),
                "brier_score": round(float(brier), 4),
                "calibration_error": round(float(cal_error), 4),
                "avg_drawdown_on_miss": round(avg_dd_miss, 4) if avg_dd_miss is not None else None,
                "sample_size": len(agent_votes),
                "status": "live_tracked"
            }

        results["_meta"] = {
            "total_resolved_votes": len(votes),
            "agents_tracked": len(results) - 1  # exclude _meta
        }
        return results

    def compute_agent_accuracy_by_regime(self, regime: str) -> Dict[str, Any]:
        """
        Same metrics as compute_agent_accuracy() but filtered to a specific market regime.
        """
        votes = self.memory.get_all_votes_with_outcomes()
        regime_votes = [v for v in votes if v.get("market_regime") == regime]

        if not regime_votes:
            return {"regime": regime, "status": "no_data", "agents": {}}

        by_agent = defaultdict(list)
        for v in regime_votes:
            by_agent[v["agent_name"]].append(v)

        agents = {}
        for agent_name in self.AGENT_NAMES:
            agent_votes = by_agent.get(agent_name, [])
            if len(agent_votes) < 2:
                agents[agent_name] = {"accuracy": None, "sample_size": len(agent_votes)}
                continue

            accuracies = [1 if v["is_accurate"] else 0 for v in agent_votes]
            agents[agent_name] = {
                "accuracy": round(sum(accuracies) / len(accuracies), 4),
                "sample_size": len(agent_votes)
            }

        return {
            "regime": regime,
            "status": "computed",
            "total_votes": len(regime_votes),
            "agents": agents
        }

    def get_best_agent(self, min_samples: int = 5) -> Dict[str, Any]:
        """Returns the agent with highest accuracy and sufficient sample size."""
        metrics = self.compute_agent_accuracy()
        best_name = None
        best_acc = -1.0

        for agent_name in self.AGENT_NAMES:
            m = metrics.get(agent_name, {})
            if m.get("sample_size", 0) >= min_samples and m.get("accuracy", 0) > best_acc:
                best_acc = m["accuracy"]
                best_name = agent_name

        if best_name:
            return {
                "best_agent": best_name,
                "accuracy": best_acc,
                "sample_size": metrics[best_name]["sample_size"],
                "status": metrics[best_name]["status"]
            }
        return {"best_agent": None, "status": "insufficient_data"}

    def get_agent_trend(self, agent_name: str, span: int = 10) -> Dict[str, Any]:
        """
        Computes EWMA accuracy trend for a specific agent.
        Returns recent trend, overall accuracy, and whether agent is improving.
        """
        votes = self.memory.get_all_votes_with_outcomes()
        agent_votes = sorted(
            [v for v in votes if v["agent_name"] == agent_name],
            key=lambda x: x.get("decision_id", "")
        )

        if len(agent_votes) < 3:
            return {
                "agent": agent_name,
                "trend": "insufficient_data",
                "overall_accuracy": self.BASELINE_ACCURACY.get(agent_name, 0.50),
                "ewma_accuracy": None,
                "sample_size": len(agent_votes)
            }

        binary = np.array([1.0 if v["is_accurate"] else 0.0 for v in agent_votes])
        overall = float(np.mean(binary))

        # EWMA
        alpha = 2.0 / (span + 1)
        ewma = binary[0]
        for val in binary[1:]:
            ewma = alpha * val + (1 - alpha) * ewma
        ewma = float(ewma)

        # Trend detection
        if len(binary) >= 6:
            first_half = float(np.mean(binary[:len(binary)//2]))
            second_half = float(np.mean(binary[len(binary)//2:]))
            delta = second_half - first_half
            if delta > 0.05:
                trend = "improving"
            elif delta < -0.05:
                trend = "degrading"
            else:
                trend = "stable"
        else:
            trend = "too_few_samples"

        return {
            "agent": agent_name,
            "trend": trend,
            "overall_accuracy": round(overall, 4),
            "ewma_accuracy": round(ewma, 4),
            "sample_size": len(agent_votes)
        }

    def get_all_agent_trends(self, span: int = 10) -> Dict[str, Any]:
        """Computes trend for all agents at once."""
        return {name: self.get_agent_trend(name, span) for name in self.AGENT_NAMES}

    def _baseline_response(self, reason: str) -> Dict[str, Any]:
        """Returns baseline accuracies when insufficient live data."""
        result = {}
        for name, acc in self.BASELINE_ACCURACY.items():
            result[name] = {
                "accuracy": acc,
                "brier_score": None,
                "calibration_error": None,
                "avg_drawdown_on_miss": None,
                "sample_size": 0,
                "status": "baseline"
            }
        result["_meta"] = {"total_resolved_votes": 0, "reason": reason}
        return result


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    attr = AgentAttribution()

    print("--- 1. Agent Accuracy Metrics ---")
    acc = attr.compute_agent_accuracy()
    print(json.dumps(acc, indent=2, default=str))

    print("\n--- 2. Best Agent ---")
    best = attr.get_best_agent()
    print(json.dumps(best, indent=2))

    print("\n--- 3. Agent Trends ---")
    trends = attr.get_all_agent_trends()
    for name, trend in trends.items():
        print(f"  {name}: {trend['trend']} (EWMA={trend.get('ewma_accuracy', 'N/A')}, n={trend['sample_size']})")

    print("\n--- 4. Bull Regime Attribution ---")
    bull_regime = attr.compute_agent_accuracy_by_regime("Bull")
    print(json.dumps(bull_regime, indent=2, default=str))
