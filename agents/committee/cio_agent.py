"""
Chief Investment Officer (CIO) Agent for Fin_Sentiment_MLPipeline.

Responsibilities:
- Presides over the Investment Committee debate
- Synthesizes arguments from Bull Analyst, Bear Analyst, Risk Officer, Evidence Prosecutor, and Portfolio Manager
- Calculates objective weighted decision score and calibrated confidence
- Resolves conflicts and issues authoritative verdict:
  [STRONG BUY, BUY, WATCHLIST, HOLD, REDUCE, SELL, REJECTED]
- Authors the executive CIO Investment Rationale
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from typing import Dict, List, Any, Optional

from committee.decision_score import DecisionScorer
from committee.governance_rules import GovernanceEngine


class ChiefInvestmentOfficer:
    """Executive decision maker synthesizing multi-agent debate into an authoritative verdict."""

    def __init__(
        self,
        scorer: Optional[DecisionScorer] = None,
        governance: Optional[GovernanceEngine] = None
    ):
        self.scorer = scorer or DecisionScorer()
        self.gov = governance or GovernanceEngine()

    def resolve_debate(
        self,
        ticker: str,
        bull_case: Dict[str, Any],
        bear_case: Dict[str, Any],
        risk_case: Dict[str, Any],
        prosecutor_case: Dict[str, Any],
        pm_case: Dict[str, Any],
        evidence: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Executes formal committee adjudication.
        """
        evidence = evidence or {}

        # 1. Compile votes for consensus calculation
        votes = [
            {"agent": "BullAnalyst", "stance": bull_case.get("stance", "HOLD")},
            {"agent": "BearAnalyst", "stance": bear_case.get("stance", "HOLD")},
            {"agent": "RiskOfficer", "stance": risk_case.get("stance", "HOLD")},
            {"agent": "EvidenceProsecutor", "stance": prosecutor_case.get("stance", "PASS")}
        ]
        consensus = self.scorer.compute_consensus(votes)

        # 2. Extract constituent scores & Evidence metrics
        bull_score = bull_case.get("bull_score", 50.0)
        bear_score = bear_case.get("bear_score", 30.0)
        risk_score = risk_case.get("risk_score", 20.0)
        evidence_score = prosecutor_case.get("evidence_score", 80.0)
        evidence_quality = prosecutor_case.get("evidence_quality", "HIGH")
        agreement_score = prosecutor_case.get("evidence_agreement_score", 0.50)
        veto_triggered = risk_case.get("veto_triggered", False)
        veto_reasons = risk_case.get("veto_reasons", [])

        ml_prob = float(evidence.get("probability", evidence.get("ml_prob", 0.50)))
        threshold = float(evidence.get("threshold", evidence.get("optimal_threshold", 0.55)))
        analogs = evidence.get("historical_analogs", evidence.get("analogs", {}))
        analog_win_rate = float(analogs.get("success_rate", evidence.get("analog_win_rate", 0.50)))
        ci = analogs.get("confidence_interval", evidence.get("confidence_interval", [0.0, 0.0]))
        ci_lower = ci[0] if len(ci) > 0 else 0.0

        # 3. Evaluate Rule-Based Governance Hierarchy
        rule_eval = self.scorer.evaluate_rules(
            bull_score=bull_score,
            bear_score=bear_score,
            risk_score=risk_score,
            evidence_score=evidence_score,
            evidence_quality=evidence_quality,
            evidence_agreement_score=agreement_score,
            model_probability=ml_prob,
            optimal_threshold=threshold,
            analog_win_rate=analog_win_rate,
            ci_lower=ci_lower,
            veto_triggered=veto_triggered,
            veto_reasons=veto_reasons
        )
        decision = rule_eval["decision"]
        category_reason = rule_eval["category_reason"]

        # 4. Calibrate Confidence
        calibrated_conf = self.scorer.calibrate_confidence(
            ml_confidence=ml_prob,
            analog_confidence=analog_win_rate,
            consensus_score=consensus,
            evidence_quality=evidence_quality
        )

        # 5. Position Sizing
        if decision in ("REJECTED", "SELL"):
            final_allocation = 0.0
        elif decision == "WATCHLIST":
            final_allocation = 0.0
        elif decision == "HOLD":
            final_allocation = pm_case.get("allocation", 0.05)
        elif decision == "REDUCE":
            final_allocation = max(0.0, pm_case.get("allocation", 0.05) * 0.5)
        elif decision == "STRONG BUY":
            final_allocation = max(pm_case.get("allocation", 0.10), 0.12)
        else:  # BUY
            final_allocation = pm_case.get("allocation", 0.08)

        # 6. Author Institutional CIO Rationale
        rationale_parts = []
        if decision == "REJECTED":
            rationale_parts.append(
                f"CIO VERDICT: PROPOSAL REJECTED. Institutional veto enforced: {category_reason}"
            )
        elif decision == "WATCHLIST":
            rationale_parts.append(
                f"CIO VERDICT: PLACED ON WATCHLIST. {category_reason} "
                f"Analog precedents show a {analog_win_rate:.0%} success rate, but ML model probability remains below threshold ({ml_prob:.1%} < {threshold:.1%}). "
                f"Confidence interval [{ci[0]:+.2%}, {ci[1]:+.2%}] crosses zero, requiring evidence confirmation before deploying capital."
            )
        elif decision in ("STRONG BUY", "BUY"):
            top_bull = bull_case.get("arguments", ["Robust quantitative tailwinds"])[0]
            rationale_parts.append(
                f"CIO VERDICT: {decision} APPROVED with {calibrated_conf:.0%} calibrated confidence. "
                f"Committee resolved in favor of capital deployment ({final_allocation:.1%} allocation). "
                f"Primary thesis: {top_bull}. Risk score is acceptable ({risk_score:.0f}/100) with {agreement_score:.0%} cross-stream agreement."
            )
        elif decision == "REDUCE":
            rationale_parts.append(
                f"CIO VERDICT: REDUCE ALLOCATION. {category_reason} "
                f"Trimming exposure to {final_allocation:.1%} to mitigate drawdown risks."
            )
        else:  # HOLD
            rationale_parts.append(
                f"CIO VERDICT: MAINTAIN HOLD. {category_reason} "
                f"Preserving existing allocation ({final_allocation:.1%}) with balanced risk-reward."
            )

        cio_rationale = " ".join(rationale_parts)

        return {
            "ticker": ticker.upper(),
            "decision": decision,
            "confidence": round(calibrated_conf, 2),
            "bull_score": int(bull_score),
            "bear_score": int(bear_score),
            "risk_score": int(risk_score),
            "evidence_score": int(evidence_score),
            "analog_win_rate": round(analog_win_rate, 2),
            "model_probability": round(ml_prob, 3),
            "evidence_quality": evidence_quality,
            "evidence_agreement_score": round(agreement_score, 2),
            "consensus_score": round(consensus, 2),
            "allocation": round(final_allocation, 3),
            "governance_passed": not veto_triggered,
            "cio_rationale": cio_rationale
        }



if __name__ == "__main__":
    cio = ChiefInvestmentOfficer()
    res = cio.resolve_debate(
        ticker="NVDA",
        bull_case={"stance": "BUY", "confidence": 0.74, "bull_score": 75.0, "arguments": ["Strong ML signal (74%)", "Robust news sentiment"]},
        bear_case={"stance": "HOLD", "confidence": 0.65, "bear_score": 42.0, "arguments": ["RSI at 71.5 indicates overbought risk"]},
        risk_case={"stance": "BUY", "risk_level": "LOW", "risk_score": 15.0, "veto_triggered": False},
        prosecutor_case={"stance": "PASS", "evidence_quality": "HIGH", "evidence_score": 90.0, "issues": []},
        pm_case={"action": "BUY", "allocation": 0.08},
        evidence={"ml_prob": 0.74, "analog_win_rate": 0.78}
    )
    import json
    print("CIO Resolution Output:")
    print(json.dumps(res, indent=2))
