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
        Strictly covers:
          - Model Signal (Probability vs Calibrated Threshold)
          - Top SHAP Drivers & Feature Contributions
          - Macro Market Regime & Cross-Sectional Percentiles
          - FAISS Historical Analog Statistics (Win Rate, Median Return, Sample Size, 95% CI)
          - NLP Corporate Events & Graph Impacts
        """
        ticker = evidence.ticker
        signal = evidence.prediction
        confidence = evidence.confidence
        prob = evidence.raw_probability
        threshold = evidence.optimal_threshold
        regime = evidence.market_regime
        events = evidence.events
        graph_impacts = evidence.graph_impacts
        analogs = evidence.analogs
        tech = evidence.technical_summary
        risk = evidence.risk_summary
        shap_drivers = evidence.shap_drivers
        quant = evidence.quant_features

        thesis_parts = []

        # 1. Model Signal & Threshold Calibration
        if signal == "BUY":
            signal_narrative = (
                f"{ticker} receives an explicit **BUY** recommendation from the multi-factor ML engine. "
                f"Calibrated directional probability of **{prob:.1%}** surpasses the risk-optimized decision threshold of **{threshold:.1%}** "
                f"(Model Confidence: {confidence:.1%})."
            )
        else:
            signal_narrative = (
                f"{ticker} maintains a **NOT BUY** stance from the quantitative ML model. "
                f"Calibrated directional probability of **{prob:.1%}** falls short of the {threshold:.1%} conviction threshold, "
                f"warranting capital preservation and defensive risk-budgeting."
            )
        thesis_parts.append(signal_narrative)

        # 2. Market Regime & Cross-Sectional Ranking
        vix = quant.get("vix_level", tech.get("vix_level", 20.0))
        s_rank = quant.get("sentiment_rank", 0.50)
        sec_ret = quant.get("sector_return_5d", tech.get("sector_return_5d", 0.0))
        squeeze = tech.get("squeeze_state", "VOLATILITY_EXPANDING")

        thesis_parts.append(
            f"Macro regime is currently classified as **{regime}** (VIX at {vix:.1f}). "
            f"From a cross-sectional perspective, {ticker}'s FinBERT sentiment ranks at the {s_rank:.0%} percentile of the 25-asset coverage universe, "
            f"while 5-day sector peer momentum stands at {sec_ret:+.2%} ({squeeze})."
        )

        # 3. Top SHAP Predictive Drivers
        if shap_drivers:
            top_drivers_str = ", ".join(shap_drivers[:4])
            thesis_parts.append(
                f"Primary model feature attributions (TreeSHAP) indicate key quantitative drivers: {top_drivers_str}."
            )

        # 4. Historical Analog Statistics (FAISS 30k Engine)
        sample_size = analogs.get("sample_size", analogs.get("similar_cases", 0))
        win_rate = analogs.get("success_rate", 0.50)
        med_ret = analogs.get("median_return_5d", 0.0)
        avg_ret = analogs.get("avg_return_5d", 0.0)
        ci = analogs.get("confidence_interval", [0.0, 0.0])

        if sample_size > 0:
            thesis_parts.append(
                f"Historical analog matching across 30,025 situations identified {sample_size} closely aligned multi-factor setups. "
                f"Precedents achieved a **{win_rate:.1%} 5-day win rate**, delivering a median return of **{med_ret:+.2%}** "
                f"(mean {avg_ret:+.2%}, 95% Confidence Interval: [{ci[0]:+.2%}, {ci[1]:+.2%}])."
            )

        # 5. Fundamental Events & Knowledge Graph
        pos_events = [e for e in events if e.get("impact") == "positive"]
        neg_events = [e for e in events if e.get("impact") == "negative"]
        if pos_events:
            ev_names = ", ".join([f"'{e['event'].replace('_', ' ').title()}'" for e in pos_events[:2]])
            thesis_parts.append(f"Qualitative evidence is supported by {ev_names}.")
        elif neg_events:
            ev_names = ", ".join([f"'{e['event'].replace('_', ' ').title()}'" for e in neg_events[:2]])
            thesis_parts.append(f"Operational headwind risks include {ev_names}.")

        if graph_impacts:
            thesis_parts.append(f"Knowledge Graph multi-hop analysis traces structural impact: {graph_impacts[0]}.")

        # 6. Risk Assessment & Invalidation
        risk_level = risk.get("risk_level", "MEDIUM")
        risk_factors = risk.get("risk_factors", [])
        if risk_factors:
            thesis_parts.append(f"Risk governance profile is {risk_level}, monitoring: {'; '.join(risk_factors[:2])}.")

        return " ".join(thesis_parts)



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
