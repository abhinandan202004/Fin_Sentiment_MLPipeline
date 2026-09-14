"""
Evidence Prosecutor Agent for Fin_Sentiment_MLPipeline.

Responsibilities:
- Adversarially audits evidence quality (HIGH, MEDIUM, LOW)
- Detects conflicting signals between ML models, sentiment, technicals, and analogs
- Flags low-confidence data (scant news articles, tiny analog sample sizes)
- Protects the Investment Committee against garbage-in, garbage-out decisions
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from typing import Dict, List, Any, Optional


class EvidenceProsecutor:
    """Audits data integrity and challenges evidence strength before debate begins."""

    def analyze(
        self,
        ticker: str,
        evidence: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Audits evidence package for completeness, conflicts, and signal strength.
        """
        evidence = evidence or {}
        issues: List[str] = []
        strengths: List[str] = []

        # 1. News Article Sample Size
        article_count = evidence.get("article_count", 5)
        if article_count < 2:
            issues.append(f"Insufficient news sample size (only {article_count} articles available)")
        elif article_count < 4:
            issues.append(f"Sparse news coverage ({article_count} articles); sentiment may be volatile")
        else:
            strengths.append(f"Adequate news coverage ({article_count} independent articles analyzed)")

        # 2. Analog Pattern Sample Size
        analog_count = evidence.get("analog_count", 10)
        analog_win_rate = evidence.get("analog_win_rate", 0.65)
        if analog_count < 3:
            issues.append(f"Analog sample size too small ({analog_count} historical analogs found)")
        elif analog_count < 6:
            issues.append(f"Limited historical analog matches ({analog_count} historical regimes)")
        else:
            strengths.append(f"Robust analog sample ({analog_count} matches with {analog_win_rate:.0%} success rate)")

        # 3. Conflict Detection (Sentiment vs Technicals vs ML)
        sentiment_score = evidence.get("sentiment_score", 0.35)
        ml_prob = evidence.get("ml_prob", 0.65)
        rsi = evidence.get("rsi", 58.0)
        macd_diff = evidence.get("macd_diff", 0.25)

        # Conflict A: ML is very bullish but FinBERT sentiment is negative
        if ml_prob > 0.60 and sentiment_score < -0.10:
            issues.append(f"Signal Divergence: ML model is bullish ({ml_prob:.0%}) while FinBERT news sentiment is negative ({sentiment_score:+.2f})")

        # Conflict B: ML is bullish but RSI is heavily overbought
        if ml_prob > 0.60 and rsi > 72.0:
            issues.append(f"Technical Discrepancy: Bullish ML prediction enters severely overbought territory (RSI {rsi:.1f} > 72)")

        # Conflict C: Bullish sentiment but MACD shows negative crossover
        if sentiment_score > 0.30 and macd_diff < -0.20:
            issues.append("Momentum Drag: Sentiment is optimistic but MACD histogram exhibits negative divergence")

        # 4. Determine Quality Rating & Evidence Score
        issue_penalty = len(issues) * 20
        base_score = 90
        evidence_score = max(min(base_score - issue_penalty, 100), 10)

        if len(issues) >= 3 or (article_count < 2 and analog_count < 3):
            quality = "LOW"
            stance = "REJECT"
        elif len(issues) >= 1:
            quality = "MEDIUM"
            stance = "PASS"
        else:
            quality = "HIGH"
            stance = "PASS"

        return {
            "agent": "EvidenceProsecutor",
            "ticker": ticker.upper(),
            "stance": stance,
            "evidence_quality": quality,
            "evidence_score": float(evidence_score),
            "confidence": 0.85 if quality == "HIGH" else (0.70 if quality == "MEDIUM" else 0.50),
            "issues": issues,
            "strengths": strengths,
            "arguments": issues if issues else ["Evidence verified: No structural data deficiencies or acute signal divergence."]
        }


if __name__ == "__main__":
    prosecutor = EvidenceProsecutor()
    print("--- 1. Testing Robust Evidence ---")
    good_ev = prosecutor.analyze("NVDA", {
        "article_count": 8,
        "analog_count": 12,
        "analog_win_rate": 0.78,
        "sentiment_score": 0.55,
        "ml_prob": 0.72,
        "rsi": 61.0,
        "macd_diff": 0.30
    })
    print("Quality:", good_ev["evidence_quality"], "| Stance:", good_ev["stance"], "| Score:", good_ev["evidence_score"])

    print("\n--- 2. Testing Low-Confidence / Conflicted Evidence ---")
    weak_ev = prosecutor.analyze("TCS", {
        "article_count": 1,
        "analog_count": 2,
        "analog_win_rate": 0.45,
        "sentiment_score": -0.25,
        "ml_prob": 0.65,
        "rsi": 74.0,
        "macd_diff": -0.15
    })
    print("Quality:", weak_ev["evidence_quality"], "| Stance:", weak_ev["stance"], "| Score:", weak_ev["evidence_score"])
    print("Issues:", weak_ev["issues"])
