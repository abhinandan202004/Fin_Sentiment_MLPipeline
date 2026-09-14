"""
Risk Officer Agent for Fin_Sentiment_MLPipeline.

Responsibilities:
- Evaluates portfolio risk, sector concentration, volatility risk, and tail risk
- Enforces institutional risk guardrails and evaluates RISK_VETO_RULES
- Holds VETO power over the investment committee
- Calculates risk_score and sets strict maximum position limits
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from typing import Dict, List, Any, Optional

from committee.governance_rules import GovernanceEngine, RISK_VETO_RULES


class RiskOfficer:
    """Independent risk arbiter with veto authority over investment proposals."""

    def __init__(self, governance_engine: Optional[GovernanceEngine] = None):
        self.gov = governance_engine or GovernanceEngine()

    def analyze(
        self,
        ticker: str,
        evidence: Optional[Dict[str, Any]] = None,
        current_portfolio: Optional[Dict[str, float]] = None,
        sector_mapping: Optional[Dict[str, str]] = None,
        risk_metrics: Optional[Dict[str, Any]] = None,
        proposed_allocation: float = 0.15,
        cash_buffer: float = 0.10
    ) -> Dict[str, Any]:
        """
        Assesses holistic risk profile and determines veto actions or position limits.
        """
        evidence = evidence or {}
        current_portfolio = current_portfolio or {}
        sector_mapping = sector_mapping or {}
        risk_metrics = risk_metrics or {}

        args: List[str] = []
        volatility = evidence.get("volatility", 0.25)
        beta = risk_metrics.get("beta", 1.05)
        var_95 = risk_metrics.get("var_95", 0.025)
        hhi = risk_metrics.get("hhi", 0.20)

        # Run compliance & veto engine
        comp = self.gov.evaluate_compliance(
            ticker=ticker,
            target_allocation=proposed_allocation,
            current_portfolio=current_portfolio,
            sector_mapping=sector_mapping,
            risk_metrics=risk_metrics,
            cash_buffer=cash_buffer
        )

        veto_triggered = not comp["governance_passed"]
        vetoes = comp["vetoes_triggered"]

        # Calculate risk score (0 to 100)
        risk_calc = 0.0
        if volatility > 0.30:
            risk_calc += (volatility - 0.30) * 150.0
            args.append(f"Elevated individual asset volatility ({volatility:.1%}) warrants strict position sizing")
        elif volatility <= 0.20:
            risk_calc -= 10.0
            args.append(f"Favorable low-volatility profile ({volatility:.1%}) within risk tolerance")

        if beta > 1.15:
            risk_calc += (beta - 1.15) * 60.0
            args.append(f"Elevated systematic risk (Portfolio Beta: {beta:.2f})")

        if hhi > 0.25:
            risk_calc += 15.0
            args.append(f"Concentration risk alert: Portfolio HHI at {hhi:.3f} exceeds balanced threshold")

        if veto_triggered:
            risk_calc += 45.0
            for v in comp["violations"]:
                args.append(f"VETO TRIGGERED: {v}")

        risk_score = round(min(max(risk_calc + 20.0, 5.0), 98.0), 1)

        # Classify risk level
        if veto_triggered or risk_score >= 60.0:
            risk_level = "CRITICAL" if veto_triggered else "HIGH"
            stance = "REJECT" if veto_triggered else "REDUCE"
            position_limit = 0.0 if veto_triggered else 0.05
        elif risk_score >= 35.0:
            risk_level = "MEDIUM"
            stance = "HOLD"
            position_limit = 0.12
        else:
            risk_level = "LOW"
            stance = "BUY"
            position_limit = min(proposed_allocation, self.gov.max_single_position)

        return {
            "agent": "RiskOfficer",
            "ticker": ticker.upper(),
            "stance": stance,
            "risk_level": risk_level,
            "risk_score": risk_score,
            "position_limit": round(position_limit, 4),
            "veto_triggered": veto_triggered,
            "veto_reasons": vetoes,
            "confidence": 0.88,
            "arguments": args
        }


if __name__ == "__main__":
    ro = RiskOfficer()
    print("--- 1. Testing Standard Risk Assessment ---")
    res1 = ro.analyze(
        ticker="NVDA",
        evidence={"volatility": 0.32},
        current_portfolio={"NVDA": 0.05, "AAPL": 0.15, "Cash": 0.15},
        sector_mapping={"NVDA": "Technology", "AAPL": "Technology"},
        risk_metrics={"beta": 1.08, "var_95": 0.026, "hhi": 0.18},
        proposed_allocation=0.12
    )
    print("Risk Level:", res1["risk_level"], "| Stance:", res1["stance"], "| Limit:", res1["position_limit"])
    for a in res1["arguments"]:
        print("  *", a)

    print("\n--- 2. Testing Veto Trigger (Sector Cap) ---")
    res2 = ro.analyze(
        ticker="NVDA",
        evidence={"volatility": 0.35},
        current_portfolio={"MSFT": 0.25, "AAPL": 0.20, "Cash": 0.10},
        sector_mapping={"NVDA": "Technology", "MSFT": "Technology", "AAPL": "Technology"},
        risk_metrics={"beta": 1.18, "var_95": 0.032, "hhi": 0.28},
        proposed_allocation=0.20
    )
    print("Risk Level:", res2["risk_level"], "| Veto:", res2["veto_triggered"], "| Stance:", res2["stance"])
    for a in res2["arguments"]:
        print("  *", a)
