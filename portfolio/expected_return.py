"""
Expected Return Engine for Fin_Sentiment_MLPipeline.

Combines:
1. Machine Learning Probability / Directional Signal
2. Historical Analog Success Rate
3. FinBERT Sentiment Score
4. Sector Relative Strength / Trend Momentum

Outputs an annualized expected return for asset allocation and Sharpe optimization.
"""

import json
from typing import Dict, Any, Optional


class ExpectedReturnEngine:
    """Calculates multi-modal expected returns combining ML, analogs, sentiment, and sector momentum."""

    def __init__(
        self,
        weight_ml: float = 0.40,
        weight_analog: float = 0.25,
        weight_sentiment: float = 0.20,
        weight_sector: float = 0.15,
        baseline_market_return: float = 0.10
    ):
        self.w_ml = weight_ml
        self.w_analog = weight_analog
        self.w_sent = weight_sentiment
        self.w_sec = weight_sector
        self.baseline_return = baseline_market_return

    def calculate_ticker_return(
        self,
        ticker: str,
        ml_prob: float = 0.60,
        analog_success_rate: float = 0.65,
        sentiment_score: float = 0.45,
        sector_strength: float = 0.08,
        historical_annual_drift: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Calculates expected return for an individual asset.

        Args:
            ticker: Stock symbol (e.g. 'NVDA')
            ml_prob: Probability of positive 5-day return from ML model [0, 1]
            analog_success_rate: Historical win rate of similar topological market setups [0, 1]
            sentiment_score: FinBERT net sentiment [-1.0, 1.0]
            sector_strength: Relative sector momentum / return [-0.20, 0.30]
            historical_annual_drift: Prior asset annualized mean return (fallback baseline)
        """
        base = historical_annual_drift if historical_annual_drift is not None else self.baseline_return

        # 1. ML Contribution: Centered around 0.5; maps [0, 1] -> [-15%, +15%] delta
        ml_delta = (ml_prob - 0.50) * 0.30

        # 2. Analog Contribution: Centered around 0.5; maps [0, 1] -> [-10%, +10%] delta
        analog_delta = (analog_success_rate - 0.50) * 0.20

        # 3. Sentiment Contribution: maps [-1, 1] -> [-8%, +8%] delta
        sent_delta = sentiment_score * 0.08

        # 4. Sector Strength: sector alpha contribution
        sector_delta = sector_strength * 0.50

        # Weighted combination
        alpha = (
            self.w_ml * ml_delta +
            self.w_analog * analog_delta +
            self.w_sent * sent_delta +
            self.w_sec * sector_delta
        )

        expected_return = round(base + alpha, 4)

        return {
            "ticker": ticker,
            "expected_return": expected_return,
            "components": {
                "base_drift": round(base, 4),
                "ml_contribution": round(self.w_ml * ml_delta, 4),
                "analog_contribution": round(self.w_analog * analog_delta, 4),
                "sentiment_contribution": round(self.w_sent * sent_delta, 4),
                "sector_contribution": round(self.w_sec * sector_delta, 4)
            }
        }

    def batch_expected_returns(
        self,
        assets_data: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Computes expected returns for a universe of assets.
        Returns {ticker: expected_return}.
        """
        results = {}
        for ticker, data in assets_data.items():
            res = self.calculate_ticker_return(
                ticker=ticker,
                ml_prob=data.get("ml_prob", 0.55),
                analog_success_rate=data.get("analog_success_rate", 0.55),
                sentiment_score=data.get("sentiment_score", 0.20),
                sector_strength=data.get("sector_strength", 0.05),
                historical_annual_drift=data.get("historical_annual_drift", None)
            )
            results[ticker] = res["expected_return"]
        return results


if __name__ == "__main__":
    engine = ExpectedReturnEngine()
    print("--- Single Asset Return ---")
    nvda_res = engine.calculate_ticker_return(
        ticker="NVDA",
        ml_prob=0.68,
        analog_success_rate=0.74,
        sentiment_score=0.62,
        sector_strength=0.12
    )
    print(json.dumps(nvda_res, indent=2))

    print("\n--- Universe Batch Returns ---")
    universe = {
        "NVDA": {"ml_prob": 0.68, "analog_success_rate": 0.74, "sentiment_score": 0.62, "sector_strength": 0.12},
        "AAPL": {"ml_prob": 0.54, "analog_success_rate": 0.58, "sentiment_score": 0.25, "sector_strength": 0.08},
        "MSFT": {"ml_prob": 0.61, "analog_success_rate": 0.62, "sentiment_score": 0.40, "sector_strength": 0.10},
        "TCS":  {"ml_prob": 0.46, "analog_success_rate": 0.48, "sentiment_score": -0.10, "sector_strength": -0.02}
    }
    batch = engine.batch_expected_returns(universe)
    print(json.dumps(batch, indent=2))
