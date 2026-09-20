"""
Bear Analyst Agent for Fin_Sentiment_MLPipeline.

Responsibilities:
- Finds reasons NOT to buy; actively searches for failure modes
- Challenges bullish assumptions and detects technical exhaustion
- Identifies valuation risks, macro headwinds, and negative analog precedent
- Produces bear_score and skeptical counterarguments
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from typing import Dict, List, Any, Optional


class BearAnalyst:
    """Builds the adversarial short/skeptical case and highlights vulnerability traps."""

    def analyze(
        self,
        ticker: str,
        evidence: Optional[Dict[str, Any]] = None,
        regime: str = "Bull"
    ) -> Dict[str, Any]:
        """
        Scans for technical exhaustion, downside catalysts, and macro friction.
        """
        evidence = evidence or {}
        args: List[str] = []

        ml_prob = float(evidence.get("probability", evidence.get("ml_prob", 0.50)))
        threshold = float(evidence.get("threshold", evidence.get("optimal_threshold", 0.55)))

        analogs = evidence.get("historical_analogs", evidence.get("analogs", {}))
        analog_rate = float(analogs.get("success_rate", evidence.get("analog_win_rate", 0.50)))
        ci = analogs.get("confidence_interval", evidence.get("confidence_interval", [0.0, 0.0]))

        tech = evidence.get("technical_summary", {})
        rsi = float(tech.get("rsi", evidence.get("rsi", 50.0)))
        volatility = float(evidence.get("volatility", 0.25))
        macd_diff = float(tech.get("macd_diff", evidence.get("macd_diff", 0.0)))

        quant = evidence.get("quant_features", {})
        vix_level = float(quant.get("vix_level", tech.get("vix_level", 20.0)))
        top_neg_shap = evidence.get("top_shap_drivers", {}).get("negative", [])

        # 1. Model Conviction Shortfall
        if ml_prob < threshold:
            args.append(f"Model Hurdle Failure: Calibrated probability ({ml_prob:.1%}) fails to cross the {threshold:.1%} conviction threshold")
        elif ml_prob < 0.50:
            args.append(f"Bearish ML drift: Negative directional probability ({ml_prob:.1%}) signals baseline capital loss")

        # 2. Historical Analog Failure Precedent & CI Risk
        if analog_rate < 0.50:
            args.append(f"Negative analog skew: Historical setups experienced a {1.0 - analog_rate:.0%} failure rate across comparable market states")
        if len(ci) == 2 and ci[0] <= 0.0:
            args.append(f"Analog tail risk: 95% Confidence Interval [{ci[0]:+.2%}, {ci[1]:+.2%}] crosses zero into negative return territory")

        # 3. Top Headwind Drivers (TreeSHAP)
        if top_neg_shap:
            neg_str = ", ".join(top_neg_shap[:2])
            args.append(f"Predictive feature drag: Quant model heavily penalized by {neg_str}")

        # 4. Technical Momentum Exhaustion & Volatility Headwinds
        if rsi >= 70.0:
            args.append(f"Overbought technical exhaustion: RSI at {rsi:.1f} signals acute mean-reversion risk")
        if macd_diff < 0.0:
            args.append("MACD structural divergence: Negative histogram spread indicates momentum decay")
        if vix_level > 22.0:
            args.append(f"Elevated macro volatility: VIX at {vix_level:.1f} threatens multiple contraction and risk-off liquidations")

        if not args:
            args.append("Valuation multiples and macro uncertainty limit directional upside.")

        # Compute quantitative bear_score (0 - 100)
        bear_calc = 20.0
        if ml_prob < threshold:
            bear_calc += (threshold - ml_prob) * 120.0
        if analog_rate < 0.55:
            bear_calc += (0.55 - analog_rate) * 80.0
        if len(ci) == 2 and ci[0] <= 0.0:
            bear_calc += 10.0
        if vix_level > 20.0:
            bear_calc += (vix_level - 20.0) * 1.5
        if rsi > 65.0:
            bear_calc += (rsi - 65.0) * 2.0

        bear_score = round(min(max(bear_calc, 10.0), 95.0), 1)
        confidence = round(min(max((bear_score / 100.0) + 0.15, 0.35), 0.90), 2)

        stance = "SELL" if bear_score >= 65.0 else ("REDUCE" if bear_score >= 45.0 else "HOLD")

        return {
            "agent": "BearAnalyst",
            "ticker": ticker.upper(),
            "stance": stance,
            "confidence": confidence,
            "bear_score": bear_score,
            "arguments": args
        }



if __name__ == "__main__":
    bear = BearAnalyst()
    res = bear.analyze("NVDA", {
        "ml_prob": 0.72,
        "sentiment_score": 0.55,
        "analog_win_rate": 0.78,
        "rsi": 71.5,
        "volatility": 0.36,
        "macd_diff": -0.05
    }, regime="Bull")
    print("Bear Analyst Evaluation:")
    print(f"Stance: {res['stance']} (Score: {res['bear_score']}, Conf: {res['confidence']})")
    for a in res["arguments"]:
        print(f"  - {a}")
