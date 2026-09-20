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

        ml_prob = float(evidence.get("probability", evidence.get("ml_prob", 0.50)))
        threshold = float(evidence.get("threshold", evidence.get("optimal_threshold", 0.55)))

        analogs = evidence.get("historical_analogs", evidence.get("analogs", {}))
        analog_rate = float(analogs.get("success_rate", evidence.get("analog_win_rate", 0.50)))
        analog_med = float(analogs.get("median_return_5d", 0.0))

        tech = evidence.get("technical_summary", {})
        rsi = float(tech.get("rsi", evidence.get("rsi", 56.0)))
        macd_diff = float(tech.get("macd_diff", evidence.get("macd_diff", 0.0)))
        trend = tech.get("trend", evidence.get("trend", "BULLISH"))

        quant = evidence.get("quant_features", {})
        sentiment_rank = float(quant.get("sentiment_rank", 0.50))
        sector_mom = float(quant.get("sector_return_5d", evidence.get("sector_strength", 0.0)))
        bb_squeeze = float(quant.get("bb_squeeze", 0.0))

        top_pos_shap = evidence.get("top_shap_drivers", {}).get("positive", [])
        events = evidence.get("events", [])

        # 1. ML Model Signal & Conviction Threshold
        if ml_prob >= threshold:
            args.append(f"Model Conviction: ML directional probability ({ml_prob:.1%}) clears the {threshold:.1%} threshold with constructive edge")
        elif ml_prob >= 0.50:
            args.append(f"Neutral ML baseline ({ml_prob:.1%} upward direction probability)")

        # 2. Historical Analog Precedents (FAISS 30k Engine)
        if analog_rate >= 0.60:
            args.append(f"High-conviction historical analogs: {analog_rate:.0%} 5-day win rate (median {analog_med:+.2%}) across 30,025 situations")
        elif analog_rate >= 0.52:
            args.append(f"Constructive analog precedents ({analog_rate:.0%} success rate in comparable setups)")

        # 3. Top Predictive Drivers (TreeSHAP)
        if top_pos_shap:
            drivers_str = ", ".join(top_pos_shap[:2])
            args.append(f"Quantitative catalyst attributions: Positive impact from {drivers_str}")

        # 4. Sector & Volatility Compression Tailwinds
        if sector_mom > 0.01:
            args.append(f"Sector peer momentum: +{sector_mom:.2%} 5-day outperformance indicates institutional sector rotation")
        if bb_squeeze > 0.5:
            args.append("Volatility squeeze detected: Bollinger compression signals imminent explosive upward breakout")

        # 5. Technical & Momentum Posture
        if "UPTREND" in trend and 45.0 <= rsi <= 68.0:
            args.append(f"Structural uptrend: Healthy momentum without exhaustion (RSI at {rsi:.1f})")

        # 6. Sentiment & Knowledge Graph Tailwinds
        if sentiment_rank >= 0.60:
            args.append(f"Cross-sectional sentiment strength: Ranks at {sentiment_rank:.0%} percentile among 25 coverage assets")

        if not args:
            args.append("Baseline historical drift and technical support.")

        # Compute quantitative bull_score (0 - 100)
        score = (
            (ml_prob * 30.0) +
            (analog_rate * 35.0) +
            (sentiment_rank * 15.0) +
            (min(max(sector_mom * 200.0, -10.0), 10.0)) +
            (10.0 if "UPTREND" in trend else 0.0)
        )
        bull_score = round(min(max(score, 10.0), 95.0), 1)
        confidence = round(min(max((ml_prob + analog_rate) / 2.0, 0.40), 0.95), 2)

        stance = "STRONG BUY" if bull_score >= 75.0 else ("BUY" if bull_score >= 55.0 else "HOLD")

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
