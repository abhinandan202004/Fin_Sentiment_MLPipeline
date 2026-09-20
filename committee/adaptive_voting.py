"""
Adaptive Voting for Sprint 7 — Committee Learning & Performance Attribution.

Replaces fixed agent weights with performance-learned weights using
exponentially-weighted moving averages (EWMA).

Key features:
  - Minimum sample threshold: 50 resolved decisions (user-specified)
  - EWMA span=10 for recency bias (recent decisions carry 3× older ones)
  - Regime-adjusted weights: agent weights shift based on per-regime accuracy
  - Weight history tracking for visualization
  - Normalized weights that sum to 1.0
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
from typing import Dict, List, Any, Optional
from collections import defaultdict

import numpy as np

from committee.committee_memory import CommitteeMemory

logger = logging.getLogger("AdaptiveVoting")

# Baseline weights from Sprint 6 (used when insufficient data)
BASELINE_WEIGHTS = {
    "BullAnalyst": 0.35,
    "BearAnalyst": 0.25,
    "RiskOfficer": 0.25,
    "EvidenceProsecutor": 0.15
}

# Minimum resolved decisions before switching to learned weights
MIN_ADAPTIVE_THRESHOLD = 50


class AdaptiveVoting:
    """Computes performance-learned agent weights using EWMA accuracy."""

    def __init__(self, min_samples: int = MIN_ADAPTIVE_THRESHOLD, ewma_span: int = 10):
        self.memory = CommitteeMemory()
        self.min_samples = min_samples
        self.ewma_span = ewma_span
        self.alpha = 2.0 / (ewma_span + 1)

    def compute_adaptive_weights(self) -> Dict[str, Any]:
        """
        Computes performance-learned agent weights.

        If resolved_count < min_samples:
            Returns baseline weights.
        If sufficient data:
            Computes EWMA accuracy per agent, normalizes to sum=1.0.

        Returns:
            {
                "weights": {"BullAnalyst": 0.30, ...},
                "status": "baseline" | "adaptive",
                "resolved_count": N,
                "agent_ewma": {"BullAnalyst": 0.65, ...}
            }
        """
        votes = self.memory.get_all_votes_with_outcomes()

        # Count unique resolved decisions
        resolved_ids = set(v["decision_id"] for v in votes)
        resolved_count = len(resolved_ids)

        if resolved_count < self.min_samples:
            return {
                "weights": BASELINE_WEIGHTS.copy(),
                "status": "baseline",
                "resolved_count": resolved_count,
                "threshold": self.min_samples,
                "reason": f"Only {resolved_count} resolved decisions, need {self.min_samples} for adaptive weights",
                "agent_ewma": {}
            }

        # Group votes by agent, sorted chronologically by decision_id
        by_agent = defaultdict(list)
        for v in sorted(votes, key=lambda x: x.get("decision_id", "")):
            by_agent[v["agent_name"]].append(v)

        # Compute EWMA accuracy per agent
        agent_ewma = {}
        for agent_name in BASELINE_WEIGHTS:
            agent_votes = by_agent.get(agent_name, [])
            if len(agent_votes) < 5:
                # Fall back to baseline for this agent
                agent_ewma[agent_name] = BASELINE_WEIGHTS[agent_name]
                continue

            binary = np.array([1.0 if v["is_accurate"] else 0.0 for v in agent_votes])
            ewma = binary[0]
            for val in binary[1:]:
                ewma = self.alpha * val + (1 - self.alpha) * ewma
            agent_ewma[agent_name] = float(ewma)

        # Normalize EWMA accuracies to sum=1.0
        total_ewma = sum(agent_ewma.values())
        if total_ewma <= 0:
            return {
                "weights": BASELINE_WEIGHTS.copy(),
                "status": "baseline",
                "resolved_count": resolved_count,
                "reason": "Zero total EWMA accuracy",
                "agent_ewma": agent_ewma
            }

        weights = {agent: round(ewma / total_ewma, 4) for agent, ewma in agent_ewma.items()}

        logger.info(
            f"Adaptive weights computed from {resolved_count} decisions: "
            + ", ".join(f"{a}={w:.2%}" for a, w in weights.items())
        )

        return {
            "weights": weights,
            "status": "adaptive",
            "resolved_count": resolved_count,
            "agent_ewma": {a: round(e, 4) for a, e in agent_ewma.items()},
            "ewma_span": self.ewma_span
        }

    def apply_regime_adjustment(
        self,
        weights: Dict[str, float],
        regime: str
    ) -> Dict[str, Any]:
        """
        Adjusts agent weights based on per-regime accuracy.

        If an agent has 80% accuracy in Bear markets but only 45% in Bull markets,
        the weights shift accordingly for the current regime.
        """
        votes = self.memory.get_all_votes_with_outcomes()
        regime_votes = [v for v in votes if v.get("market_regime") == regime]

        if len(regime_votes) < 10:
            return {
                "adjusted_weights": weights.copy(),
                "status": "unadjusted",
                "reason": f"Only {len(regime_votes)} votes in {regime} regime, need 10+"
            }

        # Compute per-agent accuracy in this regime
        by_agent = defaultdict(list)
        for v in regime_votes:
            by_agent[v["agent_name"]].append(1 if v["is_accurate"] else 0)

        regime_accuracy = {}
        for agent in weights:
            if agent in by_agent and len(by_agent[agent]) >= 3:
                regime_accuracy[agent] = sum(by_agent[agent]) / len(by_agent[agent])
            else:
                regime_accuracy[agent] = 0.50  # neutral default

        # Blend: 70% base weight + 30% regime-adjusted
        adjusted = {}
        for agent in weights:
            base = weights[agent]
            regime_factor = regime_accuracy.get(agent, 0.50)
            adjusted[agent] = base * 0.7 + regime_factor * 0.3

        # Re-normalize
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {a: round(w / total, 4) for a, w in adjusted.items()}

        return {
            "adjusted_weights": adjusted,
            "regime": regime,
            "regime_accuracy": {a: round(acc, 4) for a, acc in regime_accuracy.items()},
            "status": "regime_adjusted"
        }

    def get_weight_history(self, window_size: int = 10) -> List[Dict[str, Any]]:
        """
        Returns the evolution of weights over time using a sliding window.

        Computes weights at each step of `window_size` resolved decisions
        to show how agent weights shift as more data accumulates.
        """
        votes = self.memory.get_all_votes_with_outcomes()
        if not votes:
            return []

        # Sort chronologically
        sorted_votes = sorted(votes, key=lambda x: x.get("decision_id", ""))

        # Group by decision_id to get chronological decision order
        decision_ids = []
        seen = set()
        for v in sorted_votes:
            did = v["decision_id"]
            if did not in seen:
                decision_ids.append(did)
                seen.add(did)

        history = []
        for i in range(window_size, len(decision_ids) + 1, max(window_size // 2, 1)):
            window_dids = set(decision_ids[:i])
            window_votes = [v for v in sorted_votes if v["decision_id"] in window_dids]

            by_agent = defaultdict(list)
            for v in window_votes:
                by_agent[v["agent_name"]].append(1 if v["is_accurate"] else 0)

            accuracies = {}
            for agent in BASELINE_WEIGHTS:
                if agent in by_agent and by_agent[agent]:
                    accuracies[agent] = sum(by_agent[agent]) / len(by_agent[agent])
                else:
                    accuracies[agent] = BASELINE_WEIGHTS[agent]

            total = sum(accuracies.values())
            weights = {a: round(acc / total, 4) for a, acc in accuracies.items()} if total > 0 else BASELINE_WEIGHTS.copy()

            history.append({
                "decisions_count": i,
                "weights": weights,
                "accuracies": {a: round(acc, 4) for a, acc in accuracies.items()}
            })

        return history


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    av = AdaptiveVoting()

    print("--- 1. Adaptive Weights ---")
    result = av.compute_adaptive_weights()
    print(json.dumps(result, indent=2))

    print(f"\n--- 2. Weight Status: {result['status'].upper()} ---")
    print(f"Resolved: {result['resolved_count']} / Threshold: {av.min_samples}")
    for agent, weight in result["weights"].items():
        print(f"  {agent}: {weight:.2%}")

    print("\n--- 3. Weight History ---")
    history = av.get_weight_history()
    for h in history[-3:]:
        print(f"  @{h['decisions_count']} decisions: {h['weights']}")
