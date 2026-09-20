"""
Counterfactual Analysis for Sprint 7 — Committee Learning & Performance Attribution.

Answers the critical institutional question:
  "Is the committee adding value, or would a simpler system perform equally well?"

Three counterfactual benchmarks:
  1. Committee vs Individual Agents — Did the committee outperform any single agent?
  2. Committee vs ML Model Only — Did the committee outperform a pure ML threshold strategy?
  3. Committee vs Consensus Signals — Did the committee outperform a simple majority-vote system?

Each benchmark computes:
  - Directional accuracy
  - Average 5d/10d/20d returns
  - Average max drawdown
  - Risk-adjusted edge (return / volatility)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
from typing import Dict, List, Any, Optional
from collections import defaultdict

import numpy as np

from committee.committee_memory import CommitteeMemory

logger = logging.getLogger("Counterfactual")

DEFAULT_ML_THRESHOLD = 0.55


class CounterfactualAnalysis:
    """Measures committee value-add against simpler decision strategies."""

    def __init__(self):
        self.memory = CommitteeMemory()

    def run_full_analysis(self) -> Dict[str, Any]:
        """
        Runs all three counterfactual comparisons and returns comprehensive results.
        """
        decisions = self.memory.get_all_resolved_decisions()
        votes = self.memory.get_all_votes_with_outcomes()

        if not decisions or len(decisions) < 3:
            return {
                "status": "insufficient_data",
                "total_decisions": len(decisions) if decisions else 0,
                "reason": "Need at least 3 resolved decisions for counterfactual analysis"
            }

        committee_perf = self._compute_committee_performance(decisions)
        ml_only = self._compute_ml_only_performance(decisions)
        individual_agents = self._compute_individual_agent_performance(decisions, votes)
        consensus = self._compute_consensus_performance(decisions)

        # Compute edges
        edges = {}
        if committee_perf["accuracy"] is not None and ml_only["accuracy"] is not None:
            edges["committee_vs_ml"] = {
                "accuracy_edge": round(committee_perf["accuracy"] - ml_only["accuracy"], 4),
                "return_edge": round(
                    (committee_perf.get("avg_return_5d") or 0) - (ml_only.get("avg_return_5d") or 0), 4
                ),
                "drawdown_edge": round(
                    (committee_perf.get("avg_drawdown") or 0) - (ml_only.get("avg_drawdown") or 0), 4
                ),
                "verdict": self._compute_verdict(committee_perf, ml_only)
            }

        if committee_perf["accuracy"] is not None and consensus["accuracy"] is not None:
            edges["committee_vs_consensus"] = {
                "accuracy_edge": round(committee_perf["accuracy"] - consensus["accuracy"], 4),
                "return_edge": round(
                    (committee_perf.get("avg_return_5d") or 0) - (consensus.get("avg_return_5d") or 0), 4
                ),
                "drawdown_edge": round(
                    (committee_perf.get("avg_drawdown") or 0) - (consensus.get("avg_drawdown") or 0), 4
                ),
                "verdict": self._compute_verdict(committee_perf, consensus)
            }

        # Best individual agent
        best_agent_name = None
        best_agent_acc = -1.0
        for agent_name, perf in individual_agents.items():
            if perf.get("accuracy") is not None and perf["accuracy"] > best_agent_acc:
                best_agent_acc = perf["accuracy"]
                best_agent_name = agent_name

        if best_agent_name and committee_perf["accuracy"] is not None:
            best_agent = individual_agents[best_agent_name]
            edges["committee_vs_best_agent"] = {
                "best_agent": best_agent_name,
                "accuracy_edge": round(committee_perf["accuracy"] - best_agent["accuracy"], 4),
                "return_edge": round(
                    (committee_perf.get("avg_return_5d") or 0) - (best_agent.get("avg_return_5d") or 0), 4
                ),
                "verdict": "COMMITTEE_ADDS_VALUE" if committee_perf["accuracy"] > best_agent["accuracy"] else "AGENT_OUTPERFORMS"
            }

        return {
            "status": "computed",
            "total_decisions": len(decisions),
            "committee": committee_perf,
            "ml_only": ml_only,
            "consensus_signals": consensus,
            "individual_agents": individual_agents,
            "edges": edges,
            "overall_verdict": self._overall_verdict(edges)
        }

    def _compute_committee_performance(self, decisions: List[Dict]) -> Dict[str, Any]:
        """Computes actual committee decision performance."""
        buy_decisions = [d for d in decisions if d["decision"] in ("STRONG BUY", "BUY")]
        avoid_decisions = [d for d in decisions if d["decision"] in ("SELL", "REDUCE", "REJECTED", "HOLD", "WATCHLIST")]

        correct = 0
        total = 0
        returns_5d = []
        returns_10d = []
        returns_20d = []
        drawdowns = []

        for d in decisions:
            ret = d.get("realized_return_5d")
            if ret is None:
                continue
            total += 1
            is_positive = ret > 0

            # Committee was correct if BUY+positive or AVOID+negative
            if d["decision"] in ("STRONG BUY", "BUY") and is_positive:
                correct += 1
            elif d["decision"] in ("SELL", "REDUCE", "REJECTED", "HOLD", "WATCHLIST") and not is_positive:
                correct += 1

            returns_5d.append(ret)
            if d.get("realized_return_10d") is not None:
                returns_10d.append(d["realized_return_10d"])
            if d.get("realized_return_20d") is not None:
                returns_20d.append(d["realized_return_20d"])
            if d.get("max_drawdown_5d") is not None:
                drawdowns.append(d["max_drawdown_5d"])

        if total == 0:
            return {"accuracy": None, "sample_size": 0}

        vol = float(np.std(returns_5d)) if len(returns_5d) > 1 else None
        avg_ret = float(np.mean(returns_5d))

        return {
            "accuracy": round(correct / total, 4),
            "sample_size": total,
            "avg_return_5d": round(avg_ret, 4),
            "avg_return_10d": round(float(np.mean(returns_10d)), 4) if returns_10d else None,
            "avg_return_20d": round(float(np.mean(returns_20d)), 4) if returns_20d else None,
            "avg_drawdown": round(float(np.mean(drawdowns)), 4) if drawdowns else None,
            "volatility": round(vol, 4) if vol else None,
            "sharpe_proxy": round(avg_ret / vol, 4) if vol and vol > 0 else None,
            "buy_count": len(buy_decisions),
            "avoid_count": len(avoid_decisions)
        }

    def _compute_ml_only_performance(self, decisions: List[Dict]) -> Dict[str, Any]:
        """
        Simulates performance if only the ML model threshold was used.
        BUY if model_probability >= threshold, else AVOID.
        """
        correct = 0
        total = 0
        returns_5d = []
        drawdowns = []

        for d in decisions:
            ret = d.get("realized_return_5d")
            ml_prob = d.get("model_probability")
            if ret is None or ml_prob is None:
                continue

            total += 1
            is_positive = ret > 0
            ml_bullish = ml_prob >= DEFAULT_ML_THRESHOLD

            if (ml_bullish and is_positive) or (not ml_bullish and not is_positive):
                correct += 1

            returns_5d.append(ret)
            if d.get("max_drawdown_5d") is not None:
                drawdowns.append(d["max_drawdown_5d"])

        if total == 0:
            return {"accuracy": None, "sample_size": 0}

        vol = float(np.std(returns_5d)) if len(returns_5d) > 1 else None
        avg_ret = float(np.mean(returns_5d))

        return {
            "accuracy": round(correct / total, 4),
            "sample_size": total,
            "avg_return_5d": round(avg_ret, 4),
            "avg_drawdown": round(float(np.mean(drawdowns)), 4) if drawdowns else None,
            "volatility": round(vol, 4) if vol else None,
            "sharpe_proxy": round(avg_ret / vol, 4) if vol and vol > 0 else None,
            "strategy": "ML_THRESHOLD_ONLY"
        }

    def _compute_individual_agent_performance(
        self, decisions: List[Dict], votes: List[Dict]
    ) -> Dict[str, Any]:
        """
        Simulates performance if each individual agent made all decisions alone.
        """
        # Map decision_id -> realized_return
        decision_map = {d["id"]: d for d in decisions}

        by_agent = defaultdict(list)
        for v in votes:
            dec = decision_map.get(v["decision_id"])
            if dec and dec.get("realized_return_5d") is not None:
                by_agent[v["agent_name"]].append({
                    "stance": v["stance"],
                    "is_accurate": v["is_accurate"],
                    "realized_return_5d": dec["realized_return_5d"],
                    "max_drawdown_5d": dec.get("max_drawdown_5d")
                })

        results = {}
        for agent_name, agent_data in by_agent.items():
            if len(agent_data) < 3:
                results[agent_name] = {"accuracy": None, "sample_size": len(agent_data)}
                continue

            correct = sum(1 for d in agent_data if d["is_accurate"])
            returns = [d["realized_return_5d"] for d in agent_data]
            drawdowns = [d["max_drawdown_5d"] for d in agent_data if d.get("max_drawdown_5d") is not None]

            vol = float(np.std(returns)) if len(returns) > 1 else None
            avg_ret = float(np.mean(returns))

            results[agent_name] = {
                "accuracy": round(correct / len(agent_data), 4),
                "sample_size": len(agent_data),
                "avg_return_5d": round(avg_ret, 4),
                "avg_drawdown": round(float(np.mean(drawdowns)), 4) if drawdowns else None,
                "volatility": round(vol, 4) if vol else None,
                "sharpe_proxy": round(avg_ret / vol, 4) if vol and vol > 0 else None
            }

        return results

    def _compute_consensus_performance(self, decisions: List[Dict]) -> Dict[str, Any]:
        """
        Simulates performance if a simple majority of evidence signals was used.
        BUY if 3+ of 4 signals agree bullish, else AVOID.
        """
        correct = 0
        total = 0
        returns_5d = []
        drawdowns = []

        for d in decisions:
            ret = d.get("realized_return_5d")
            if ret is None:
                continue

            # Count bullish signals
            bullish = 0
            signal_count = 0

            ml_prob = d.get("model_probability")
            if ml_prob is not None:
                bullish += 1 if ml_prob >= DEFAULT_ML_THRESHOLD else 0
                signal_count += 1

            wr = d.get("analog_win_rate")
            if wr is not None:
                bullish += 1 if wr >= 0.55 else 0
                signal_count += 1

            bs = d.get("bull_score")
            brs = d.get("bear_score")
            if bs is not None and brs is not None:
                bullish += 1 if bs > brs else 0
                signal_count += 1

            eas = d.get("evidence_agreement_score")
            if eas is not None:
                bullish += 1 if eas > 0.50 else 0
                signal_count += 1

            if signal_count < 2:
                continue

            total += 1
            is_positive = ret > 0
            consensus_bullish = bullish >= (signal_count / 2)

            if (consensus_bullish and is_positive) or (not consensus_bullish and not is_positive):
                correct += 1

            returns_5d.append(ret)
            if d.get("max_drawdown_5d") is not None:
                drawdowns.append(d["max_drawdown_5d"])

        if total == 0:
            return {"accuracy": None, "sample_size": 0}

        vol = float(np.std(returns_5d)) if len(returns_5d) > 1 else None
        avg_ret = float(np.mean(returns_5d))

        return {
            "accuracy": round(correct / total, 4),
            "sample_size": total,
            "avg_return_5d": round(avg_ret, 4),
            "avg_drawdown": round(float(np.mean(drawdowns)), 4) if drawdowns else None,
            "volatility": round(vol, 4) if vol else None,
            "sharpe_proxy": round(avg_ret / vol, 4) if vol and vol > 0 else None,
            "strategy": "CONSENSUS_MAJORITY_VOTE"
        }

    @staticmethod
    def _compute_verdict(committee: Dict, benchmark: Dict) -> str:
        """Determines if committee adds value vs benchmark."""
        c_acc = committee.get("accuracy", 0) or 0
        b_acc = benchmark.get("accuracy", 0) or 0
        c_ret = committee.get("avg_return_5d", 0) or 0
        b_ret = benchmark.get("avg_return_5d", 0) or 0
        c_dd = committee.get("avg_drawdown", 0) or 0
        b_dd = benchmark.get("avg_drawdown", 0) or 0

        # Committee wins if: better accuracy AND (better returns OR less drawdown)
        if c_acc > b_acc and (c_ret > b_ret or c_dd > b_dd):
            return "COMMITTEE_ADDS_VALUE"
        elif c_acc > b_acc:
            return "COMMITTEE_MARGINALLY_BETTER"
        elif c_acc == b_acc:
            return "NO_DIFFERENCE"
        else:
            return "BENCHMARK_OUTPERFORMS"

    @staticmethod
    def _overall_verdict(edges: Dict) -> str:
        """Summarizes whether the committee is worth the complexity."""
        wins = 0
        losses = 0
        for comparison, edge in edges.items():
            verdict = edge.get("verdict", "")
            if "ADDS_VALUE" in verdict or "MARGINALLY_BETTER" in verdict:
                wins += 1
            elif "OUTPERFORMS" in verdict:
                losses += 1

        if wins >= 2:
            return "COMMITTEE_JUSTIFIED — Committee demonstrably outperforms simpler strategies"
        elif wins == 1 and losses == 0:
            return "COMMITTEE_MARGINAL — Committee adds marginal value; monitor closely"
        elif losses >= 2:
            return "COMMITTEE_QUESTIONABLE — Simpler strategies outperform; consider simplification"
        else:
            return "INCONCLUSIVE — Insufficient evidence to determine committee value-add"


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    cf = CounterfactualAnalysis()

    print("--- Counterfactual Analysis ---")
    result = cf.run_full_analysis()
    print(json.dumps(result, indent=2, default=str))

    if result["status"] == "computed":
        print(f"\n=== OVERALL VERDICT ===")
        print(f"  {result['overall_verdict']}")

        if "edges" in result:
            print(f"\n=== EDGES ===")
            for name, edge in result["edges"].items():
                print(f"  {name}: acc_edge={edge.get('accuracy_edge', 'N/A')}, verdict={edge.get('verdict', 'N/A')}")
