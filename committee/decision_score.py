"""
Decision Scoring & Consensus Calibration Engine for Fin_Sentiment_MLPipeline.

Implements:
1. Weighted Voting Formula:
   final_score = (0.35 * bull_score) - (0.25 * bear_score) - (0.25 * risk_score) + (0.15 * evidence_score)
2. Calibrated Confidence Metric:
   confidence = ((ml_confidence + analog_confidence + consensus_score) / 3) * quality_factor
3. Committee Consensus Metric:
   consensus_score measuring directional agreement across committee agents [0.0, 1.0]
4. Institutional Decision Categorization:
   STRONG BUY, BUY, WATCHLIST, HOLD, REDUCE, SELL, REJECTED
"""

from typing import Dict, List, Any, Optional


class DecisionScorer:
    """Calculates weighted committee decision scores, consensus, and calibrated confidence."""

    def __init__(
        self,
        weight_bull: float = 0.35,
        weight_bear: float = 0.25,
        weight_risk: float = 0.25,
        weight_evidence: float = 0.15
    ):
        self.w_bull = weight_bull
        self.w_bear = weight_bear
        self.w_risk = weight_risk
        self.w_evi = weight_evidence

    def calculate_score(
        self,
        bull_score: float,
        bear_score: float,
        risk_score: float,
        evidence_score: float,
        veto_triggered: bool = False,
        evidence_quality: str = "HIGH"
    ) -> Dict[str, Any]:
        """
        Computes final governance score and categorizes the decision.
        Scores are expected on a 0-100 scale.
        """
        if veto_triggered:
            return {
                "decision": "REJECTED",
                "final_score": -99.0,
                "bull_score": bull_score,
                "bear_score": bear_score,
                "risk_score": risk_score,
                "evidence_score": evidence_score,
                "category_reason": "Hard risk veto triggered by Risk Officer / Governance Engine."
            }

        final_score = round(
            (self.w_bull * bull_score) -
            (self.w_bear * bear_score) -
            (self.w_risk * risk_score) +
            (self.w_evi * evidence_score),
            2
        )

        quality_upper = evidence_quality.upper()

        # Decision category mapping
        if quality_upper == "LOW" and final_score >= 15.0:
            decision = "WATCHLIST"
            reason = "Promising upside signal, but evidence quality is LOW. Deferring to Watchlist until verified."
        elif final_score >= 35.0:
            decision = "STRONG BUY"
            reason = f"High conviction consensus (Final Score {final_score:.1f} >= 35.0)."
        elif final_score >= 25.0:
            decision = "BUY"
            reason = f"Positive risk-adjusted committee alignment (Final Score {final_score:.1f} >= 25.0)."
        elif final_score >= 15.0:
            decision = "WATCHLIST"
            reason = f"Moderate score ({final_score:.1f}) qualifies for priority monitoring."
        elif final_score >= 5.0:
            decision = "HOLD"
            reason = f"Neutral risk-reward tradeoff (Final Score {final_score:.1f})."
        elif final_score >= -10.0:
            decision = "REDUCE"
            reason = f"Downside risks outbalance upside drivers (Final Score {final_score:.1f} < 5.0)."
        else:
            decision = "SELL"
            reason = f"Severe bearish conviction or risk penalty (Final Score {final_score:.1f} < -10.0)."

        return {
            "decision": decision,
            "final_score": final_score,
            "bull_score": round(bull_score, 1),
            "bear_score": round(bear_score, 1),
            "risk_score": round(risk_score, 1),
            "evidence_score": round(evidence_score, 1),
            "category_reason": reason
        }

    def compute_consensus(self, votes: List[Dict[str, Any]]) -> float:
        """
        Computes consensus score [0.0, 1.0] across voting members.
        Bull/Risk positive vote counts as positive, Bear/Prosecutor negative counts as negative.
        """
        if not votes:
            return 0.50

        positive_count = 0
        total_valid = 0

        for v in votes:
            stance = v.get("stance", "").upper()
            if stance in ("BUY", "STRONG BUY", "PASS", "LOW_RISK", "ACCEPT"):
                positive_count += 1
                total_valid += 1
            elif stance in ("SELL", "REDUCE", "REJECT", "HIGH_RISK", "FAIL"):
                total_valid += 1
            elif stance in ("HOLD", "WATCHLIST", "NEUTRAL", "MEDIUM_RISK"):
                positive_count += 0.5
                total_valid += 1

        if total_valid == 0:
            return 0.50

        consensus = round(positive_count / total_valid, 2)
        return consensus

    def calibrate_confidence(
        self,
        ml_confidence: float,
        analog_confidence: float,
        consensus_score: float,
        evidence_quality: str = "HIGH"
    ) -> float:
        """
        Calibrates composite confidence across ML model, analogs, committee consensus,
        and evidence quality.
        """
        raw_conf = (ml_confidence + analog_confidence + consensus_score) / 3.0

        quality_multipliers = {
            "HIGH": 1.0,
            "MEDIUM": 0.90,
            "LOW": 0.75
        }
        multiplier = quality_multipliers.get(evidence_quality.upper(), 0.90)

        calibrated = min(max(raw_conf * multiplier, 0.10), 0.99)
        return round(calibrated, 2)


if __name__ == "__main__":
    scorer = DecisionScorer()
    print("--- 1. Testing Strong Buy Score ---")
    score_res = scorer.calculate_score(
        bull_score=84.0,
        bear_score=42.0,
        risk_score=10.0,
        evidence_score=85.0,
        evidence_quality="HIGH"
    )
    print(score_res)

    print("\n--- 2. Testing Consensus & Calibration ---")
    sample_votes = [
        {"agent": "BullAnalyst", "stance": "BUY"},
        {"agent": "BearAnalyst", "stance": "HOLD"},
        {"agent": "RiskOfficer", "stance": "BUY"},
        {"agent": "EvidenceProsecutor", "stance": "PASS"}
    ]
    consensus = scorer.compute_consensus(sample_votes)
    print("Consensus Score:", consensus)

    conf = scorer.calibrate_confidence(
        ml_confidence=0.76,
        analog_confidence=0.78,
        consensus_score=consensus,
        evidence_quality="HIGH"
    )
    print("Calibrated Confidence:", conf)
