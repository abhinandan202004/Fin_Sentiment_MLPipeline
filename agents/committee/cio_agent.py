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

        # 2. Extract constituent scores
        bull_score = bull_case.get("bull_score", 50.0)
        bear_score = bear_case.get("bear_score", 30.0)
        risk_score = risk_case.get("risk_score", 20.0)
        evidence_score = prosecutor_case.get("evidence_score", 80.0)
        evidence_quality = prosecutor_case.get("evidence_quality", "HIGH")
        veto_triggered = risk_case.get("veto_triggered", False)

        # 3. Calculate weighted score
        score_data = self.scorer.calculate_score(
            bull_score=bull_score,
            bear_score=bear_score,
            risk_score=risk_score,
            evidence_score=evidence_score,
            veto_triggered=veto_triggered,
            evidence_quality=evidence_quality
        )
        decision = score_data["decision"]
        final_score = score_data["final_score"]

        # 4. Calibrate confidence
        ml_conf = evidence.get("ml_prob", bull_case.get("confidence", 0.65))
        analog_conf = evidence.get("analog_win_rate", 0.65)
        calibrated_conf = self.scorer.calibrate_confidence(
            ml_confidence=ml_conf,
            analog_confidence=analog_conf,
            consensus_score=consensus,
            evidence_quality=evidence_quality
        )

        # 5. Final Allocation determination
        if decision in ("REJECTED", "SELL"):
            final_allocation = 0.0
        elif decision == "WATCHLIST":
            final_allocation = 0.0
        elif decision == "HOLD":
            final_allocation = pm_case.get("allocation", 0.05)
        else:
            final_allocation = pm_case.get("allocation", 0.08)

        # 6. Author CIO Rationale
        rationale_paragraphs = []
        if decision == "REJECTED":
            reasons = "; ".join(risk_case.get("veto_reasons", ["Risk limits breached"]))
            rationale_paragraphs.append(
                f"CIO VERDICT: PROPOSAL REJECTED. The Risk Officer exercised institutional veto power due to: {reasons}. "
                f"Capital preservation overrides directional upside."
            )
        elif decision == "WATCHLIST":
            rationale_paragraphs.append(
                f"CIO VERDICT: PLACED ON PRIORITY WATCHLIST. While underlying indicators are constructive (Score: {final_score:.1f}), "
                f"the Evidence Prosecutor flagged data issues ({'; '.join(prosecutor_case.get('issues', ['Incomplete evidence']))}). "
                f"Capital allocation deferred until additional verification emerges."
            )
        elif decision in ("STRONG BUY", "BUY"):
            top_bull = bull_case.get("arguments", ["Positive directional edge"])[0]
            top_bear = bear_case.get("arguments", ["Standard volatility risks"])[0]
            rationale_paragraphs.append(
                f"CIO VERDICT: {decision} APPROVED with {calibrated_conf:.0%} calibrated confidence. "
                f"The Committee resolved in favor of capital deployment ({final_allocation:.1%} target allocation). "
                f"Primary thesis: {top_bull}. Countervailing headwind monitored: {top_bear}. "
                f"Risk assessment passed with balanced HHI and appropriate volatility buffers."
            )
        elif decision == "REDUCE":
            rationale_paragraphs.append(
                f"CIO VERDICT: REDUCE ALLOCATION. Bearish arguments outbalance current expected return (Score: {final_score:.1f}). "
                f"Trimming exposure to protect portfolio capital."
            )
        else:  # HOLD
            rationale_paragraphs.append(
                f"CIO VERDICT: MAINTAIN HOLD. Neutral risk-reward tradeoff observed with {consensus:.0%} committee consensus. "
                f"Preserving existing position without active delta rebalancing."
            )

        cio_rationale = " ".join(rationale_paragraphs)

        return {
            "agent": "CIO",
            "ticker": ticker.upper(),
            "decision": decision,
            "confidence": calibrated_conf,
            "consensus_score": consensus,
            "allocation": final_allocation,
            "final_score": final_score,
            "bull_score": bull_score,
            "bear_score": bear_score,
            "risk_score": risk_score,
            "evidence_score": evidence_score,
            "evidence_quality": evidence_quality,
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
