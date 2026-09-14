"""
Bull Analyst Agent for Fin_Sentiment_MLPipeline.

Responsibilities:
- Synthesizes all bullish evidence (ML confidence, FinBERT sentiment, analog setups, technical momentum)
- Identifies commercial catalysts & knowledge graph supply-chain tailwinds
- Defends the long thesis during investment committee debate
- Produces bull_score and structured arguments
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from typing import Dict, List, Any, Optional


class BullAnalyst:
    """Builds the comprehensive bull case and defends buying opportunities."""

    def analyze(
        self,
        ticker: str,
        evidence: Optional[Dict[str, Any]] = None,
        regime: str = "Bull"
    ) -> Dict[str, Any]:
        """
        Evaluates upside potential and structures arguments in favor of capital allocation.
        """
        evidence = evidence or {}
        args: List[str] = []

        ml_prob = evidence.get("ml_prob", 0.65)
        sentiment = evidence.get("sentiment_score", 0.40)
        analog_rate = evidence.get("analog_win_rate", 0.68)
        sector_mom = evidence.get("sector_strength", 0.08)
        rsi = evidence.get("rsi", 56.0)
        events = evidence.get("events", [])

        # 1. ML Model Signal
        if ml_prob >= 0.65:
            args.append(f"Strong ML predictive signal ({ml_prob:.0%} upward direction probability)")
        elif ml_prob >= 0.52:
            args.append(f"Moderate positive ML signal ({ml_prob:.0%} probability)")

        # 2. FinBERT Sentiment & News
        if sentiment >= 0.30:
            args.append(f"Robust institutional news sentiment (FinBERT compound score: +{sentiment:.2f})")
        elif sentiment > 0.05:
            args.append(f"Constructive news flow (FinBERT score: +{sentiment:.2f})")

        # 3. Historical Analog Pattern
        if analog_rate >= 0.65:
            args.append(f"High-conviction historical analogs ({analog_rate:.0%} win rate in topologically identical setups)")
        elif analog_rate >= 0.52:
            args.append(f"Favorable historical analogs ({analog_rate:.0%} success rate)")

        # 4. Sector & Macro Regime
        if regime in ("Bull", "Low Volatility"):
            args.append(f"Macro tailwind: {regime} regime supports equity risk expansion")
        if sector_mom > 0.04:
            args.append(f"Leading sector relative strength (+{sector_mom:.1%} outperformance vs market)")

        # 5. Technical Momentum
        if 45.0 <= rsi <= 68.0:
            args.append(f"Healthy technical momentum without exhaustion (RSI at {rsi:.1f})")

        # 6. Corporate & Market Events
        for ev in events[:2]:
            event_type = ev.get("event_type", "Catalyst")
            impact = ev.get("sentiment_label", "Positive")
            args.append(f"Key catalyst: {event_type} event detected ({impact} market impact)")

        if not args:
            args.append("Stable technical foundation and baseline expected drift.")

        # Compute quantitative bull_score (0 - 100)
        score = (
            (ml_prob * 35.0) +
            (max(sentiment, 0.0) * 25.0) +
            (analog_rate * 25.0) +
            (15.0 if regime == "Bull" else 8.0)
        )
        bull_score = round(min(max(score, 10.0), 98.0), 1)
        confidence = round(min(max((ml_prob + analog_rate) / 2.0, 0.40), 0.95), 2)

        stance = "STRONG BUY" if bull_score >= 82.0 else ("BUY" if bull_score >= 60.0 else "HOLD")

        return {
            "agent": "BullAnalyst",
            "ticker": ticker.upper(),
            "stance": stance,
            "confidence": confidence,
            "bull_score": bull_score,
            "arguments": args
        }


if __name__ == "__main__":
    bull = BullAnalyst()
    res = bull.analyze("NVDA", {
        "ml_prob": 0.74,
        "sentiment_score": 0.58,
        "analog_win_rate": 0.78,
        "sector_strength": 0.12,
        "rsi": 62.0,
        "events": [{"event_type": "Product Launch", "sentiment_label": "Bullish"}]
    }, regime="Bull")
    print("Bull Analyst Evaluation:")
    print(f"Stance: {res['stance']} (Score: {res['bull_score']}, Conf: {res['confidence']})")
    for a in res["arguments"]:
        print(f"  + {a}")
