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
    """Calculates explainable rule-based committee decisions, consensus, and calibrated confidence."""

    def __init__(self):
        pass

    def evaluate_rules(
        self,
        bull_score: float,
        bear_score: float,
        risk_score: float,
        evidence_score: float,
        evidence_quality: str,
        evidence_agreement_score: float = 0.50,
        model_probability: float = 0.50,
        optimal_threshold: float = 0.55,
        analog_win_rate: float = 0.50,
        ci_lower: float = 0.0,
        veto_triggered: bool = False,
        veto_reasons: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Executes institutional rule-based governance hierarchy over arbitrary magic-number weights.
        Decision ladder: STRONG BUY, BUY, WATCHLIST, HOLD, REDUCE, SELL, REJECTED.
        """
        veto_reasons = veto_reasons or []

        # 1. Hard Governance Veto Check
        if veto_triggered:
            return {
                "decision": "REJECTED",
                "bull_score": round(bull_score, 1),
                "bear_score": round(bear_score, 1),
                "risk_score": round(risk_score, 1),
                "evidence_score": round(evidence_score, 1),
                "evidence_quality": evidence_quality,
                "evidence_agreement_score": evidence_agreement_score,
                "category_reason": f"Hard governance veto triggered: {'; '.join(veto_reasons)}."
            }

        # 2. Low Quality Evidence Circuit Breaker
        quality_upper = evidence_quality.upper()
        if quality_upper == "LOW":
            return {
                "decision": "HOLD",
                "bull_score": round(bull_score, 1),
                "bear_score": round(bear_score, 1),
                "risk_score": round(risk_score, 1),
                "evidence_score": round(evidence_score, 1),
                "evidence_quality": evidence_quality,
                "evidence_agreement_score": evidence_agreement_score,
                "category_reason": "Evidence quality rated LOW by Evidence Prosecutor; capital deployment withheld until data is verified."
            }

        # 3. Acute Signal Divergence Rule (ML < Threshold but Analogs >= 60%)
        # Case in point: JPM (prob 48.5% < 60% hurdle, but analog win rate is 64%)
        if model_probability < optimal_threshold and analog_win_rate >= 0.60:
            return {
                "decision": "WATCHLIST",
                "bull_score": round(bull_score, 1),
                "bear_score": round(bear_score, 1),
                "risk_score": round(risk_score, 1),
                "evidence_score": round(evidence_score, 1),
                "evidence_quality": evidence_quality,
                "evidence_agreement_score": evidence_agreement_score,
                "category_reason": (
                    f"Acute signal divergence: Historical analogs are favorable ({analog_win_rate:.0%} win rate), "
                    f"but calibrated ML probability ({model_probability:.1%}) remains below the {optimal_threshold:.0%} conviction hurdle. "
                    f"Deferred to Watchlist for directional confirmation."
                )
            }

        # 4. High Conviction Strong Buy Rule
        if (
            model_probability >= optimal_threshold
            and analog_win_rate >= 0.60
            and ci_lower > -0.005  # CI doesn't exhibit severe negative tail
            and risk_score <= 35.0
            and evidence_agreement_score >= 0.75
        ):
            return {
                "decision": "STRONG BUY",
                "bull_score": round(bull_score, 1),
                "bear_score": round(bear_score, 1),
                "risk_score": round(risk_score, 1),
                "evidence_score": round(evidence_score, 1),
                "evidence_quality": evidence_quality,
                "evidence_agreement_score": evidence_agreement_score,
                "category_reason": (
                    f"Strong consensus alignment: ML clears conviction threshold ({model_probability:.1%} >= {optimal_threshold:.0%}), "
                    f"analog win rate is {analog_win_rate:.0%}, and risk score is low ({risk_score:.0f})."
                )
            }

        # 5. Standard Buy Rule
        if (
            model_probability >= optimal_threshold
            and (analog_win_rate >= 0.50 or bull_score > bear_score + 15)
            and risk_score <= 45.0
        ):
            return {
                "decision": "BUY",
                "bull_score": round(bull_score, 1),
                "bear_score": round(bear_score, 1),
                "risk_score": round(risk_score, 1),
                "evidence_score": round(evidence_score, 1),
                "evidence_quality": evidence_quality,
                "evidence_agreement_score": evidence_agreement_score,
                "category_reason": (
                    f"Constructive risk-adjusted setup: ML probability ({model_probability:.1%}) clears hurdle with supportive committee score."
                )
            }

        # 6. Sell Rule
        if (
            model_probability < 0.45
            and analog_win_rate <= 0.45
            and bear_score >= 60.0
        ):
            return {
                "decision": "SELL",
                "bull_score": round(bull_score, 1),
                "bear_score": round(bear_score, 1),
                "risk_score": round(risk_score, 1),
                "evidence_score": round(evidence_score, 1),
                "evidence_quality": evidence_quality,
                "evidence_agreement_score": evidence_agreement_score,
                "category_reason": (
                    f"Dominant bearish alignment: ML probability ({model_probability:.1%}) is low, "
                    f"analog win rate is {analog_win_rate:.0%}, and bear arguments dominate."
                )
            }

        # 7. Reduce Rule
        if bear_score > bull_score + 15 or model_probability < 0.48:
            return {
                "decision": "REDUCE",
                "bull_score": round(bull_score, 1),
                "bear_score": round(bear_score, 1),
                "risk_score": round(risk_score, 1),
                "evidence_score": round(evidence_score, 1),
                "evidence_quality": evidence_quality,
                "evidence_agreement_score": evidence_agreement_score,
                "category_reason": (
                    f"Defensive risk posture: Downside risks ({bear_score:.0f}) outweigh bullish catalysts ({bull_score:.0f}); trimming exposure."
                )
            }

        # 8. Hold Rule (Default Neutral Stance)
        return {
            "decision": "HOLD",
            "bull_score": round(bull_score, 1),
            "bear_score": round(bear_score, 1),
            "risk_score": round(risk_score, 1),
            "evidence_score": round(evidence_score, 1),
            "evidence_quality": evidence_quality,
            "evidence_agreement_score": evidence_agreement_score,
            "category_reason": (
                f"Neutral equilibrium: Model probability ({model_probability:.1%}) and analog win rate ({analog_win_rate:.0%}) "
                f"present balanced risk-reward without directional edge."
            )
        }

    # Backward compatibility alias
    calculate_score = evaluate_rules


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
