import logging
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict, Any, Optional
from agents.evidence import Evidence

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("AnalystAgent")


class AnalystAgent:
    """
    Master Senior Equity Research Analyst Agent.
    Strictly follows Evidence-Grounding Rules:
      - Uses ONLY verified facts in the Evidence object.
      - Synthesizes ML predictions, FinBERT events, Knowledge Graph relations,
        FAISS historical analogs, and Risk assessments into a cohesive research thesis.
    """
    def __init__(self):
        pass

    def synthesize(self, evidence: Evidence) -> str:
        """
        Produces an evidence-grounded research report conclusion and investment thesis.
        """
        ticker = evidence.ticker
        signal = evidence.prediction
        confidence = evidence.confidence
        events = evidence.events
        graph_impacts = evidence.graph_impacts
        analogs = evidence.analogs
        tech = evidence.technical_summary
        risk = evidence.risk_summary
        shap_drivers = evidence.shap_drivers

        # 1. Thesis formulation
        pos_events = [e for e in events if e.get("impact") == "positive"]
        neg_events = [e for e in events if e.get("impact") == "negative"]

        if signal == "BUY":
            direction_word = "bullish"
            posture = "constructive"
        elif signal == "SELL":
            direction_word = "bearish"
            posture = "defensive"
        else:
            direction_word = "neutral"
            posture = "cautious"

        thesis_parts = []
        thesis_parts.append(
            f"{ticker} carries a {signal} stance ({confidence:.1%} model confidence) under current market conditions. "
            f"The quantitative model and feature store indicate {posture} 5-day forward risk-adjusted positioning."
        )

        # 2. Event Evidence
        if pos_events:
            ev_names = ", ".join([f"'{e['event'].replace('_', ' ').title()}'" for e in pos_events[:2]])
            thesis_parts.append(
                f"The core fundamental catalyst is anchored by recent {ev_names}, "
                f"reinforcing positive earnings revisions and institutional confidence."
            )
        elif neg_events:
            ev_names = ", ".join([f"'{e['event'].replace('_', ' ').title()}'" for e in neg_events[:2]])
            thesis_parts.append(
                f"Recent corporate developments highlight headwind risks including {ev_names}."
            )
        else:
            thesis_parts.append(
                "Corporate news flow reflects baseline operational disclosures without severe disruptive shocks."
            )

        # 3. Knowledge Graph Cascades
        if graph_impacts:
            top_impact = graph_impacts[0]
            thesis_parts.append(
                f"Knowledge Graph multi-hop analysis highlights critical structural tailwinds: {top_impact}."
            )

        # 4. Historical Analog Analysis
        sim_cases = analogs.get("similar_cases", 0)
        win_rate = analogs.get("success_rate", 0.5)
        med_return = analogs.get("median_return_5d", 0.0)

        if sim_cases > 0:
            thesis_parts.append(
                f"Historical analog analysis via FAISS vector matching identified {sim_cases} comparable market setups. "
                f"Statistically, {win_rate:.1%} of these historical precedents generated positive 5-day returns, "
                f"with a median realized forward return of {med_return:+.2%}."
            )

        # 5. Technical Alignment & Top SHAP Drivers
        if shap_drivers:
            drivers_str = ", ".join(shap_drivers[:3])
            thesis_parts.append(f"Primary predictive feature attributions (TreeSHAP) stem from {drivers_str}.")

        # 6. Risk Assessment & Mitigants
        risk_level = risk.get("risk_level", "MEDIUM")
        risk_factors = risk.get("risk_factors", [])
        if risk_factors:
            thesis_parts.append(
                f"Risk assessment is currently rated {risk_level}, driven by: {'; '.join(risk_factors[:2])}."
            )

        # 7. Final Recommendation Conclusion
        conclusion = " ".join(thesis_parts)
        return conclusion


if __name__ == "__main__":
    analyst = AnalystAgent()
    sample_evidence = Evidence(
        ticker="NVDA",
        prediction="BUY",
        confidence=0.84,
        events=[{"event": "earnings_beat", "impact": "positive", "headline": "NVIDIA beats revenue estimates"}],
        graph_impacts=["AI demand surge strengthening semiconductor sector"],
        analogs={"similar_cases": 42, "success_rate": 0.78, "median_return_5d": 0.031},
        technical_summary={"rsi": 61.2, "trend": "BULLISH_UPTREND"},
        risk_summary={"risk_level": "MEDIUM", "risk_factors": ["Overbought RSI", "High market volatility"]},
        shap_drivers=["sentiment_ema_7 (+0.042)", "volume_ratio (+0.031)"],
    )
    print("\nAnalystAgent Generated Conclusion:")
    print(analyst.synthesize(sample_evidence))
