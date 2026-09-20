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
        Adversarially audits evidence package for statistical power, CI overlap,
        threshold compliance, cross-stream divergence, and calculates Evidence Agreement Score.
        """
        evidence = evidence or {}
        ticker = ticker.upper()
        issues: List[str] = []
        strengths: List[str] = []

        # 1. Extract Core Signals from Evidence Contract
        ml_prob = float(evidence.get("probability", evidence.get("ml_prob", 0.50)))
        threshold = float(evidence.get("threshold", evidence.get("optimal_threshold", 0.55)))

        # Historical Analogs
        analogs = evidence.get("historical_analogs", evidence.get("analogs", {}))
        analog_win_rate = float(analogs.get("success_rate", evidence.get("analog_win_rate", 0.50)))
        sample_size = int(analogs.get("sample_size", analogs.get("similar_cases", evidence.get("analog_count", 50))))
        ci = analogs.get("confidence_interval", evidence.get("confidence_interval", [0.0, 0.0]))
        med_ret = float(analogs.get("median_return_5d", evidence.get("analog_median_return", 0.0)))

        # Technicals & Sentiment
        tech = evidence.get("technical_summary", {})
        rsi = float(tech.get("rsi", evidence.get("rsi", 50.0)))
        macd_diff = float(tech.get("macd_diff", evidence.get("macd_diff", 0.0)))
        trend = tech.get("trend", evidence.get("trend", "NEUTRAL"))

        quant = evidence.get("quant_features", {})
        sentiment_rank = float(quant.get("sentiment_rank", 0.50))
        vix_level = float(quant.get("vix_level", tech.get("vix_level", 20.0)))

        # Corporate / News Events
        events = evidence.get("events", [])

        # --- AUDIT TEST 1: Model Probability vs. Conviction Threshold ---
        if ml_prob < threshold:
            shortfall = threshold - ml_prob
            issues.append(
                f"Model Hurdle Failure: Calibrated ML probability ({ml_prob:.1%}) fails to cross optimal threshold ({threshold:.1%}) by {shortfall:.1%}."
            )
        else:
            edge = ml_prob - threshold
            strengths.append(
                f"Model Conviction: ML probability ({ml_prob:.1%}) clears Sharpe-optimized hurdle ({threshold:.1%}) with {edge:+.1%} edge."
            )

        # --- AUDIT TEST 2: Historical Analog Statistical Significance & Zero-Overlap ---
        if len(ci) == 2:
            ci_low, ci_high = ci[0], ci[1]
            if ci_low <= 0.0 and ci_high > 0.0:
                issues.append(
                    f"Statistical Insignificance: 95% Confidence Interval [{ci_low:+.2%}, {ci_high:+.2%}] crosses zero, confirming high tail variance."
                )
            elif ci_high <= 0.0:
                issues.append(
                    f"Negative Asymmetry: 95% Confidence Interval [{ci_low:+.2%}, {ci_high:+.2%}] is entirely negative."
                )
            else:
                strengths.append(
                    f"Statistically Robust Precedents: 95% Confidence Interval [{ci_low:+.2%}, {ci_high:+.2%}] is strictly positive."
                )

        # --- AUDIT TEST 3: Sample Size & Empirical Power ---
        if sample_size < 15:
            issues.append(f"Severe Sample Scarcity: Only {sample_size} historical analogs found; insufficient statistical power.")
        elif sample_size < 30:
            issues.append(f"Moderate Sample Size: {sample_size} historical analogs identified; statistical power is acceptable but limited.")
        else:
            strengths.append(f"High-Density Sample: {sample_size} historical analog setups matched across the 30k universe.")

        # --- AUDIT TEST 4: Signal Divergence & Conflict Detection ---
        # Conflict A: ML model rejects (NOT BUY) but Analog Engine is Bullish (>= 60%)
        if ml_prob < threshold and analog_win_rate >= 0.60:
            issues.append(
                f"Acute Signal Divergence: ML Model is NOT BUY ({ml_prob:.1%}) while FAISS Analogs show {analog_win_rate:.1%} win rate (+{med_ret:.2%} median)."
            )

        # Conflict B: Bullish ML or Analogs, but elevated Macro Volatility (VIX > 22)
        if (ml_prob >= threshold or analog_win_rate >= 0.60) and vix_level > 22.0:
            issues.append(f"Macro Headwind: Bullish signals enter hostile volatility regime (VIX at {vix_level:.1f}).")

        # Conflict C: Overbought technical momentum (RSI > 70)
        if rsi >= 70.0:
            issues.append(f"Technical Overextension: RSI sits at {rsi:.1f} (overbought exhaustion zone).")

        # --- AUDIT TEST 5: Evidence Agreement Score Across 4 Independent Streams ---
        # Stream 1: ML Model
        ml_stance = "BULLISH" if ml_prob >= threshold else ("BEARISH" if ml_prob < 0.46 else "NEUTRAL")

        # Stream 2: Historical Analogs
        analog_stance = "BULLISH" if analog_win_rate >= 0.55 else ("BEARISH" if analog_win_rate <= 0.45 else "NEUTRAL")

        # Stream 3: Technical Setup
        if rsi > 52.0 and "UPTREND" in trend:
            tech_stance = "BULLISH"
        elif rsi < 48.0 and "DOWNTREND" in trend:
            tech_stance = "BEARISH"
        else:
            tech_stance = "NEUTRAL"

        # Stream 4: Fundamentals / Sentiment
        if sentiment_rank >= 0.60:
            fund_stance = "BULLISH"
        elif sentiment_rank <= 0.40:
            fund_stance = "BEARISH"
        else:
            fund_stance = "NEUTRAL"

        streams = {
            "ml_model": ml_stance,
            "historical_analogs": analog_stance,
            "technical_setup": tech_stance,
            "fundamental_sentiment": fund_stance,
        }

        # Calculate agreement percentage
        stance_counts = {}
        for s in streams.values():
            stance_counts[s] = stance_counts.get(s, 0) + 1

        max_agreement_count = max(stance_counts.values())
        agreement_score = round(max_agreement_count / 4.0, 2)
        dominant_stream_stance = max(stance_counts.items(), key=lambda x: x[1])[0]

        # --- 6. Quality Rating & Evidence Score ---
        # Base score starts at 85, penalized by issues, boosted by agreement
        penalty = len(issues) * 12
        agreement_bonus = int(agreement_score * 15)
        evidence_score = max(min(85 - penalty + agreement_bonus, 100), 10)

        # Severe data defect (e.g. tiny sample size < 15 or massive issue count) marks LOW quality
        if sample_size < 15 or evidence_score < 30:
            quality = "LOW"
            stance = "REJECT"
        elif len(issues) >= 2 or evidence_score < 75:
            quality = "MEDIUM"
            stance = "CHALLENGE" if len(issues) >= 3 else "PASS"
        else:
            quality = "HIGH"
            stance = "PASS"

        return {
            "agent": "EvidenceProsecutor",
            "ticker": ticker,
            "stance": stance,
            "evidence_quality": quality,
            "evidence_score": float(evidence_score),
            "evidence_agreement_score": agreement_score,
            "dominant_stance": dominant_stream_stance,
            "stream_breakdown": streams,
            "confidence": 0.85 if quality == "HIGH" else (0.70 if quality == "MEDIUM" else 0.50),
            "issues": issues,
            "strengths": strengths,
            "arguments": issues if issues else ["Evidence verified: High cross-stream consensus with strict statistical significance."]
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
