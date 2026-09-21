"""
Opportunity Scoring Engine for Sprint 8: Autonomous Research & Market Opportunity Discovery.

Computes a calibrated, explainable multi-factor Opportunity Score in [0, 100]
and a confidence metric in [0, 1.0] for any given asset in the coverage universe.

Factors:
  1. ML Conviction (30%): Distance of calibrated probability from decision hurdle
  2. Historical Analogs (25%): 30k FAISS analog win rate & sample power
  3. Sentiment Rank (15%): Cross-sectional sentiment percentile across the universe
  4. Sector & Technical Momentum (15%): 5d sector return, RSI, and squeeze state
  5. Event Impact (15%): NLP event detection and corporate announcement magnitude

Adjusted by macro regime multipliers (Bull, Bear, Sideways, High/Low Volatility).
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("OpportunityScoring")


class OpportunityScorer:
    """Calculates multi-factor opportunity scores and confidence levels."""

    DEFAULT_WEIGHTS = {
        "ml": 0.30,
        "analog": 0.25,
        "sentiment": 0.15,
        "momentum": 0.15,
        "event": 0.15
    }

    REGIME_MULTIPLIERS = {
        "Bull": 1.10,
        "Low Volatility": 1.05,
        "Neutral": 1.00,
        "Sideways": 0.95,
        "High Volatility": 0.88,
        "Bear": 0.85
    }

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        # Ensure weights sum to 1.0
        total_w = sum(self.weights.values())
        if abs(total_w - 1.0) > 1e-4:
            self.weights = {k: v / total_w for k, v in self.weights.items()}

    def score_ml(self, prob: float, threshold: float = 0.55) -> float:
        """
        Scores ML model conviction in [0, 100].
        At hurdle (prob == threshold) -> score = 60.
        Above hurdle -> 60 to 100.
        Below hurdle -> 0 to 59.
        """
        prob = max(0.0, min(1.0, float(prob)))
        threshold = max(0.40, min(0.70, float(threshold)))

        if prob >= threshold:
            # Scale from 60 to 100 based on headroom
            headroom = (prob - threshold) / max(0.01, 1.0 - threshold)
            return min(100.0, 60.0 + headroom * 40.0)
        else:
            # Scale from 0 to 59
            ratio = prob / max(0.01, threshold)
            return max(0.0, ratio * 59.0)

    def score_analogs(self, win_rate: float, sample_size: int = 50, ci_lower: float = 0.0) -> float:
        """
        Scores FAISS historical analog performance in [0, 100].
        Win rate >= 50% -> 50 to 100, penalized if sample size is small (< 30).
        """
        win_rate = max(0.0, min(1.0, float(win_rate)))
        base_score = win_rate * 100.0

        # Sample power adjustment
        sample_penalty = 0.0
        if sample_size < 15:
            sample_penalty = 15.0
        elif sample_size < 30:
            sample_penalty = 7.0

        # CI penalty if lower bound is deeply negative
        ci_penalty = 0.0
        if ci_lower < -0.02:
            ci_penalty = 8.0
        elif ci_lower < 0.0:
            ci_penalty = 3.0

        return max(0.0, min(100.0, base_score - sample_penalty - ci_penalty))

    def score_sentiment(self, rank: float, raw_sentiment: float = 0.0) -> float:
        """
        Scores sentiment percentile in [0, 100].
        Rank is percentile (0.0 to 1.0).
        """
        rank = max(0.0, min(1.0, float(rank)))
        return rank * 100.0

    def score_momentum(
        self,
        sector_return_5d: float = 0.0,
        rsi: float = 50.0,
        squeeze_state: str = "VOLATILITY_EXPANDING"
    ) -> float:
        """
        Scores technical and sector momentum in [0, 100].
        RSI in sweet spot [45, 65] with positive sector return scores highest.
        """
        # RSI score (peak around 55-62 for constructive momentum without overbought)
        if 45.0 <= rsi <= 65.0:
            rsi_score = 75.0 + (rsi - 45.0)  # 75 - 95
        elif rsi > 75.0:
            rsi_score = 40.0  # Overbought penalty
        elif rsi < 30.0:
            rsi_score = 55.0  # Oversold bounce potential
        else:
            rsi_score = 60.0

        # Sector return score: centered at 0% -> 50, +5% -> 90, -5% -> 10
        sec_score = 50.0 + (sector_return_5d * 800.0)
        sec_score = max(0.0, min(100.0, sec_score))

        # Squeeze bonus
        squeeze_bonus = 5.0 if "SQUEEZE" in squeeze_state.upper() else 0.0

        momentum_score = 0.60 * sec_score + 0.40 * rsi_score + squeeze_bonus
        return max(0.0, min(100.0, momentum_score))

    def score_events(self, impact_score: float = 0.0, has_recent_catalyst: bool = False) -> float:
        """
        Scores corporate event impact in [0, 100].
        Impact is typically [-1.0, 1.0], centered at 0 -> 50.
        """
        event_score = 50.0 + (impact_score * 40.0)
        if has_recent_catalyst:
            event_score += 10.0
        return max(0.0, min(100.0, event_score))

    def compute_opportunity(
        self,
        ticker: str,
        ml_prob: float,
        ml_threshold: float = 0.55,
        analog_win_rate: float = 0.50,
        analog_sample_size: int = 50,
        analog_ci_lower: float = 0.0,
        sentiment_rank: float = 0.50,
        raw_sentiment: float = 0.0,
        sector_return_5d: float = 0.0,
        rsi: float = 50.0,
        squeeze_state: str = "VOLATILITY_EXPANDING",
        event_impact: float = 0.0,
        has_recent_catalyst: bool = False,
        market_regime: str = "Neutral"
    ) -> Dict[str, Any]:
        """
        Computes composite Opportunity Score and confidence for a ticker.
        """
        s_ml = self.score_ml(ml_prob, ml_threshold)
        s_analog = self.score_analogs(analog_win_rate, analog_sample_size, analog_ci_lower)
        s_sent = self.score_sentiment(sentiment_rank, raw_sentiment)
        s_mom = self.score_momentum(sector_return_5d, rsi, squeeze_state)
        s_event = self.score_events(event_impact, has_recent_catalyst)

        # Weighted raw composite
        raw_score = (
            self.weights["ml"] * s_ml +
            self.weights["analog"] * s_analog +
            self.weights["sentiment"] * s_sent +
            self.weights["momentum"] * s_mom +
            self.weights["event"] * s_event
        )

        # Apply regime multiplier
        regime_mult = self.REGIME_MULTIPLIERS.get(market_regime, 1.00)
        # Multiplier adjusts distance from baseline 50
        adjusted_score = 50.0 + (raw_score - 50.0) * regime_mult
        opportunity_score = max(0.0, min(100.0, round(adjusted_score, 1)))

        # Compute confidence metric based on signal agreement
        subscores = [s_ml, s_analog, s_sent, s_mom, s_event]
        # Standard deviation of subscores: lower dispersion = higher agreement
        import numpy as np
        std_dev = float(np.std(subscores))
        # Agreement score: 1.0 when std_dev is 0, ~0.6 when std_dev is 25
        agreement = max(0.30, min(1.0, 1.0 - (std_dev / 50.0)))
        confidence = round(0.50 * agreement + 0.50 * (opportunity_score / 100.0), 2)

        return {
            "ticker": ticker.upper(),
            "opportunity_score": opportunity_score,
            "confidence": confidence,
            "subscores": {
                "ml": round(s_ml, 1),
                "analog": round(s_analog, 1),
                "sentiment": round(s_sent, 1),
                "momentum": round(s_mom, 1),
                "event": round(s_event, 1)
            },
            "market_regime": market_regime,
            "regime_multiplier": regime_mult
        }
