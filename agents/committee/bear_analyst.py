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

        ml_prob = evidence.get("ml_prob", 0.55)
        sentiment = evidence.get("sentiment_score", 0.10)
        analog_rate = evidence.get("analog_win_rate", 0.55)
        rsi = evidence.get("rsi", 62.0)
        volatility = evidence.get("volatility", 0.28)
        macd_diff = evidence.get("macd_diff", 0.10)

        # 1. Technical Exhaustion & Overbought Metrics
        if rsi >= 70.0:
            args.append(f"Severely overbought momentum (RSI at {rsi:.1f} > 70.0), high mean-reversion risk")
        elif rsi >= 65.0:
            args.append(f"Elevated technical levels (RSI at {rsi:.1f}), upside momentum slowing")

        if macd_diff < 0:
            args.append("MACD histogram indicates weakening bullish momentum and negative divergence")

        # 2. Volatility & Risk Drag
        if volatility >= 0.35:
            args.append(f"Extremely high annualized volatility ({volatility:.1%}) elevates downside tail risk")
        elif volatility >= 0.25:
            args.append(f"Above-average asset volatility ({volatility:.1%}) introduces execution friction")

        # 3. Weak or Fragile ML/Sentiment Support
        if ml_prob < 0.55:
            args.append(f"Weak ML forward edge ({ml_prob:.0%} probability fails to clear high-conviction hurdle)")

        if sentiment < 0.0:
            args.append(f"Negative news sentiment drag (FinBERT score: {sentiment:+.2f})")
        elif sentiment < 0.20:
            args.append("Tepid news sentiment provides insufficient institutional impulse")

        # 4. Analog Downside Precedent
        if analog_rate < 0.55:
            args.append(f"Historical analog setups experienced {1.0 - analog_rate:.0%} failure rate")

        # 5. Macro Regime Headwinds
        if regime in ("Bear", "High Volatility"):
            args.append(f"Macro headwind: {regime} regime threatens multiple contraction")
        elif regime == "Sideways":
            args.append("Choppy sideways macro backdrop limits trend extension")

        if not args:
            args.append("Valuation multiples remain stretched relative to broader market averages.")

        # Compute quantitative bear_score (0 - 100)
        bear_calc = 0.0
        if rsi > 65:
            bear_calc += (rsi - 65) * 2.5
        if volatility > 0.25:
            bear_calc += (volatility - 0.25) * 120.0
        if ml_prob < 0.60:
            bear_calc += (0.60 - ml_prob) * 100.0
        if sentiment < 0.20:
            bear_calc += max(0.20 - sentiment, 0.0) * 50.0
        if regime in ("Bear", "High Volatility"):
            bear_calc += 25.0

        bear_score = round(min(max(bear_calc + 15.0, 5.0), 95.0), 1)
        confidence = round(min(max(bear_score / 100.0 + 0.20, 0.35), 0.90), 2)

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
