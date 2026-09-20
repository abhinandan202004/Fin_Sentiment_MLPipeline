"""
Evidence Attribution for Sprint 7 — Committee Learning & Performance Attribution.

Tracks which evidence streams correctly predicted market outcomes:
  - ML_MODEL: Did model_probability >= threshold predict positive 5d return?
  - ANALOG_ENGINE: Did analog_win_rate >= 0.55 predict positive 5d return?
  - TECHNICAL: Did the technical stance (RSI/MACD-derived) align with outcome?
  - SENTIMENT: Did avg_sentiment > 0 predict positive 5d return?

Measures:
  - Per-stream accuracy / hit rate / sample size
  - Agreement value: Do multi-stream agreements outperform disagreements?
  - Regime-stratified signal reliability
  - Drawdown attribution per signal stream
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
from typing import Dict, List, Any, Optional
from collections import defaultdict

import numpy as np

from committee.committee_memory import CommitteeMemory

logger = logging.getLogger("EvidenceAttribution")

# Default threshold for ML model conviction
DEFAULT_ML_THRESHOLD = 0.55


class EvidenceAttribution:
    """Measures reliability of each independent evidence stream."""

    SIGNAL_STREAMS = ["ML_MODEL", "ANALOG_ENGINE", "TECHNICAL", "SENTIMENT"]

    def __init__(self):
        self.memory = CommitteeMemory()

    def compute_signal_reliability(self) -> Dict[str, Any]:
        """
        Computes accuracy, hit rate, and sample size for each evidence stream.

        For each resolved decision:
          - ML_MODEL: Correct if (prob >= threshold AND return > 0) OR (prob < threshold AND return <= 0)
          - ANALOG_ENGINE: Correct if (win_rate >= 0.55 AND return > 0) OR (win_rate < 0.55 AND return <= 0)
          - TECHNICAL: Derived from bull_score vs bear_score alignment with return direction
          - SENTIMENT: Derived from evidence_agreement_score proxy (> 0.5 = bullish)
        """
        decisions = self.memory.get_all_resolved_decisions()
        if not decisions:
            return {s: {"accuracy": None, "sample_size": 0, "status": "no_data"} for s in self.SIGNAL_STREAMS}

        stream_results = {s: {"correct": 0, "total": 0, "drawdowns_on_miss": []} for s in self.SIGNAL_STREAMS}

        for dec in decisions:
            ret_5d = dec.get("realized_return_5d")
            if ret_5d is None:
                continue

            is_positive = ret_5d > 0.0
            dd = dec.get("max_drawdown_5d")

            # ML Model signal
            ml_prob = dec.get("model_probability")
            if ml_prob is not None:
                ml_bullish = ml_prob >= DEFAULT_ML_THRESHOLD
                ml_correct = (ml_bullish and is_positive) or (not ml_bullish and not is_positive)
                stream_results["ML_MODEL"]["total"] += 1
                if ml_correct:
                    stream_results["ML_MODEL"]["correct"] += 1
                elif dd is not None:
                    stream_results["ML_MODEL"]["drawdowns_on_miss"].append(dd)

            # Analog Engine signal
            analog_wr = dec.get("analog_win_rate")
            if analog_wr is not None:
                analog_bullish = analog_wr >= 0.55
                analog_correct = (analog_bullish and is_positive) or (not analog_bullish and not is_positive)
                stream_results["ANALOG_ENGINE"]["total"] += 1
                if analog_correct:
                    stream_results["ANALOG_ENGINE"]["correct"] += 1
                elif dd is not None:
                    stream_results["ANALOG_ENGINE"]["drawdowns_on_miss"].append(dd)

            # Technical signal (derived from bull_score vs bear_score)
            bull_s = dec.get("bull_score")
            bear_s = dec.get("bear_score")
            if bull_s is not None and bear_s is not None:
                tech_bullish = bull_s > bear_s
                tech_correct = (tech_bullish and is_positive) or (not tech_bullish and not is_positive)
                stream_results["TECHNICAL"]["total"] += 1
                if tech_correct:
                    stream_results["TECHNICAL"]["correct"] += 1
                elif dd is not None:
                    stream_results["TECHNICAL"]["drawdowns_on_miss"].append(dd)

            # Sentiment signal (evidence_agreement_score > 0.5 implies bullish consensus)
            eas = dec.get("evidence_agreement_score")
            if eas is not None:
                sent_bullish = eas > 0.50
                sent_correct = (sent_bullish and is_positive) or (not sent_bullish and not is_positive)
                stream_results["SENTIMENT"]["total"] += 1
                if sent_correct:
                    stream_results["SENTIMENT"]["correct"] += 1
                elif dd is not None:
                    stream_results["SENTIMENT"]["drawdowns_on_miss"].append(dd)

        # Compile results
        results = {}
        for stream in self.SIGNAL_STREAMS:
            sr = stream_results[stream]
            if sr["total"] == 0:
                results[stream] = {"accuracy": None, "sample_size": 0, "status": "no_data"}
                continue

            accuracy = sr["correct"] / sr["total"]
            avg_dd_miss = float(np.mean(sr["drawdowns_on_miss"])) if sr["drawdowns_on_miss"] else None

            results[stream] = {
                "accuracy": round(accuracy, 4),
                "hit_rate": round(accuracy, 4),
                "correct": sr["correct"],
                "incorrect": sr["total"] - sr["correct"],
                "sample_size": sr["total"],
                "avg_drawdown_on_miss": round(avg_dd_miss, 4) if avg_dd_miss is not None else None,
                "status": "computed"
            }

        results["_meta"] = {"total_decisions": len(decisions)}
        return results

    def compute_signal_agreement_value(self) -> Dict[str, Any]:
        """
        Measures whether decisions where multiple evidence streams agree
        produce better returns than decisions where streams disagree.

        Agreement = ML, Analogs, Technical, Sentiment all directionally aligned.
        """
        decisions = self.memory.get_all_resolved_decisions()
        if not decisions:
            return {"status": "no_data"}

        agree_returns = []
        disagree_returns = []
        agree_drawdowns = []
        disagree_drawdowns = []

        for dec in decisions:
            ret_5d = dec.get("realized_return_5d")
            if ret_5d is None:
                continue

            # Count bullish signals
            signals = 0
            total_signals = 0

            ml_prob = dec.get("model_probability")
            if ml_prob is not None:
                signals += 1 if ml_prob >= DEFAULT_ML_THRESHOLD else 0
                total_signals += 1

            analog_wr = dec.get("analog_win_rate")
            if analog_wr is not None:
                signals += 1 if analog_wr >= 0.55 else 0
                total_signals += 1

            bull_s = dec.get("bull_score")
            bear_s = dec.get("bear_score")
            if bull_s is not None and bear_s is not None:
                signals += 1 if bull_s > bear_s else 0
                total_signals += 1

            eas = dec.get("evidence_agreement_score")
            if eas is not None:
                signals += 1 if eas > 0.50 else 0
                total_signals += 1

            if total_signals < 2:
                continue

            agreement_ratio = signals / total_signals
            dd = dec.get("max_drawdown_5d", 0.0) or 0.0

            # Full agreement (>= 75% of signals same direction) vs disagreement
            if agreement_ratio >= 0.75 or agreement_ratio <= 0.25:
                agree_returns.append(ret_5d)
                agree_drawdowns.append(dd)
            else:
                disagree_returns.append(ret_5d)
                disagree_drawdowns.append(dd)

        if not agree_returns or not disagree_returns:
            return {
                "status": "insufficient_data",
                "agreement_count": len(agree_returns),
                "disagreement_count": len(disagree_returns)
            }

        return {
            "agreement": {
                "count": len(agree_returns),
                "avg_return_5d": round(float(np.mean(agree_returns)), 4),
                "median_return_5d": round(float(np.median(agree_returns)), 4),
                "avg_drawdown": round(float(np.mean(agree_drawdowns)), 4),
                "positive_rate": round(sum(1 for r in agree_returns if r > 0) / len(agree_returns), 4)
            },
            "disagreement": {
                "count": len(disagree_returns),
                "avg_return_5d": round(float(np.mean(disagree_returns)), 4),
                "median_return_5d": round(float(np.median(disagree_returns)), 4),
                "avg_drawdown": round(float(np.mean(disagree_drawdowns)), 4),
                "positive_rate": round(sum(1 for r in disagree_returns if r > 0) / len(disagree_returns), 4)
            },
            "agreement_edge": round(
                float(np.mean(agree_returns)) - float(np.mean(disagree_returns)), 4
            ),
            "status": "computed"
        }

    def get_best_signal(self, regime: Optional[str] = None) -> Dict[str, Any]:
        """Returns the most reliable signal stream, optionally filtered by regime."""
        if regime:
            decisions = self.memory.get_decisions_by_regime(regime)
        else:
            decisions = self.memory.get_all_resolved_decisions()

        if not decisions:
            return {"best_signal": None, "status": "no_data"}

        # Simplified accuracy computation for each stream
        stream_acc = {}
        for stream in self.SIGNAL_STREAMS:
            correct = 0
            total = 0
            for dec in decisions:
                ret_5d = dec.get("realized_return_5d")
                if ret_5d is None:
                    continue
                is_positive = ret_5d > 0
                bullish = self._is_stream_bullish(stream, dec)
                if bullish is None:
                    continue
                total += 1
                if (bullish and is_positive) or (not bullish and not is_positive):
                    correct += 1

            if total >= 3:
                stream_acc[stream] = {"accuracy": correct / total, "sample_size": total}

        if not stream_acc:
            return {"best_signal": None, "status": "insufficient_data"}

        best = max(stream_acc, key=lambda s: stream_acc[s]["accuracy"])
        return {
            "best_signal": best,
            "accuracy": round(stream_acc[best]["accuracy"], 4),
            "sample_size": stream_acc[best]["sample_size"],
            "regime": regime,
            "status": "computed"
        }

    @staticmethod
    def _is_stream_bullish(stream: str, dec: Dict[str, Any]) -> Optional[bool]:
        """Determines if a given stream was bullish for a decision."""
        if stream == "ML_MODEL":
            p = dec.get("model_probability")
            return p >= DEFAULT_ML_THRESHOLD if p is not None else None
        elif stream == "ANALOG_ENGINE":
            wr = dec.get("analog_win_rate")
            return wr >= 0.55 if wr is not None else None
        elif stream == "TECHNICAL":
            b = dec.get("bull_score")
            br = dec.get("bear_score")
            return b > br if (b is not None and br is not None) else None
        elif stream == "SENTIMENT":
            eas = dec.get("evidence_agreement_score")
            return eas > 0.50 if eas is not None else None
        return None


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    attr = EvidenceAttribution()

    print("--- 1. Signal Reliability ---")
    rel = attr.compute_signal_reliability()
    print(json.dumps(rel, indent=2, default=str))

    print("\n--- 2. Agreement Value ---")
    av = attr.compute_signal_agreement_value()
    print(json.dumps(av, indent=2, default=str))

    print("\n--- 3. Best Signal ---")
    best = attr.get_best_signal()
    print(json.dumps(best, indent=2))
