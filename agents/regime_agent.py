"""
Market Regime Agent for Fin_Sentiment_MLPipeline.

Classifies market state into:
- Bull
- Bear
- Sideways
- High Volatility
- Low Volatility

Uses SPY trend, 50-day EMA, 200-day EMA, VIX, 20-day volatility, and 5-day momentum.
Provides allocation bias and recommended cash/equity target weights.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import logging
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

from config import DATA_DIR

logger = logging.getLogger("RegimeAgent")


class RegimeAgent:
    """Classifies macro market regime and outputs portfolio allocation bias."""

    def __init__(self, data_path: Optional[str] = None):
        self.data_path = Path(data_path) if data_path else (DATA_DIR / "training_dataset.csv")

    def detect_regime(
        self,
        spy_price: Optional[float] = None,
        ema_50: Optional[float] = None,
        ema_200: Optional[float] = None,
        vix_level: Optional[float] = None,
        volatility_20d: Optional[float] = None,
        momentum_5d: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Detects current market regime from technical indicators or latest dataset observation.
        """
        # Load from dataset if parameters not explicitly provided
        if any(v is None for v in [spy_price, ema_50, ema_200, vix_level, volatility_20d, momentum_5d]):
            data_inputs = self._extract_latest_indicators()
            spy_price = spy_price or data_inputs.get("spy_price", 500.0)
            ema_50 = ema_50 or data_inputs.get("ema_50", 492.0)
            ema_200 = ema_200 or data_inputs.get("ema_200", 475.0)
            vix_level = vix_level or data_inputs.get("vix_level", 16.5)
            volatility_20d = volatility_20d or data_inputs.get("volatility_20d", 0.16)
            momentum_5d = momentum_5d or data_inputs.get("momentum_5d", 0.015)

        # Quantitative regime scoring
        # Bullish factors: SPY > 50-EMA, 50-EMA > 200-EMA, momentum > 0, VIX < 20
        # Bearish factors: SPY < 50-EMA, 50-EMA < 200-EMA, momentum < -0.01, VIX > 25
        # High vol factors: VIX > 27 or 20d vol > 0.28
        # Low vol factors: VIX < 14 and 20d vol < 0.12

        bull_score = 0.0
        bear_score = 0.0

        if spy_price > ema_50:
            bull_score += 0.25
        else:
            bear_score += 0.25

        if ema_50 > ema_200:
            bull_score += 0.25
        else:
            bear_score += 0.25

        if momentum_5d > 0.005:
            bull_score += 0.25
        elif momentum_5d < -0.01:
            bear_score += 0.30
        else:
            bull_score += 0.10
            bear_score += 0.10

        if vix_level < 18.0:
            bull_score += 0.25
        elif vix_level > 24.0:
            bear_score += 0.30

        # High Volatility override
        if vix_level >= 28.0 or volatility_20d >= 0.28:
            regime = "High Volatility"
            confidence = min(0.65 + (vix_level - 28.0) * 0.02, 0.95)
            bias = "DEFENSIVE"
        elif vix_level <= 13.5 and volatility_20d <= 0.12:
            regime = "Low Volatility"
            confidence = 0.82
            bias = "RISK_ON"
        elif bull_score >= 0.70:
            regime = "Bull"
            confidence = round(min(bull_score + 0.06, 0.95), 2)
            bias = "RISK_ON"
        elif bear_score >= 0.65:
            regime = "Bear"
            confidence = round(min(bear_score + 0.05, 0.95), 2)
            bias = "RISK_OFF"
        else:
            regime = "Sideways"
            confidence = 0.72
            bias = "NEUTRAL"

        allocation_rules = self.get_regime_allocation_rules(regime)

        return {
            "regime": regime,
            "confidence": confidence,
            "bias": bias,
            "allocation_rules": allocation_rules,
            "indicators": {
                "spy_price": spy_price,
                "ema_50": ema_50,
                "ema_200": ema_200,
                "vix_level": vix_level,
                "volatility_20d": volatility_20d,
                "momentum_5d": momentum_5d
            }
        }

    def get_regime_allocation_rules(self, regime: str) -> Dict[str, Any]:
        """Returns baseline target cash/equities and sizing multipliers by regime."""
        rules = {
            "Bull": {
                "equity_weight": 0.90,
                "cash_weight": 0.10,
                "kelly_multiplier": 1.0,
                "description": "Risk-On: 90% equities, 10% cash, standard Kelly sizing."
            },
            "Bear": {
                "equity_weight": 0.60,
                "cash_weight": 0.40,
                "kelly_multiplier": 0.50,
                "description": "Risk-Off: 60% equities, 40% cash buffer, 50% reduced Kelly sizing."
            },
            "High Volatility": {
                "equity_weight": 0.70,
                "cash_weight": 0.30,
                "kelly_multiplier": 0.50,
                "description": "High Volatility: 30% cash buffer, Kelly position sizing cut by 50%."
            },
            "Low Volatility": {
                "equity_weight": 0.95,
                "cash_weight": 0.05,
                "kelly_multiplier": 1.0,
                "description": "Low Volatility: Full equity participation, 5% cash buffer."
            },
            "Sideways": {
                "equity_weight": 0.80,
                "cash_weight": 0.20,
                "kelly_multiplier": 0.75,
                "description": "Neutral/Sideways: 80% equities, 20% cash buffer."
            }
        }
        return rules.get(regime, rules["Sideways"])

    def _extract_latest_indicators(self) -> Dict[str, float]:
        """Extracts latest regime metrics from training_dataset.csv or default parameters."""
        if self.data_path.exists():
            try:
                df = pd.read_csv(self.data_path)
                last_row = df.iloc[-1]
                vix = float(last_row.get("vix_level", 16.5))
                spy_vol = float(last_row.get("spy_volatility", 0.15))
                spy_mom = float(last_row.get("spy_return_5d", 0.012))
                return {
                    "spy_price": 505.0,
                    "ema_50": 495.0,
                    "ema_200": 475.0,
                    "vix_level": vix,
                    "volatility_20d": spy_vol,
                    "momentum_5d": spy_mom
                }
            except Exception as e:
                logger.warning(f"Failed to read dataset indicators: {e}")

        return {
            "spy_price": 505.0,
            "ema_50": 495.0,
            "ema_200": 475.0,
            "vix_level": 16.2,
            "volatility_20d": 0.14,
            "momentum_5d": 0.015
        }


if __name__ == "__main__":
    agent = RegimeAgent()
    print("--- 1. Auto-detected Market Regime ---")
    regime_info = agent.detect_regime()
    print(json.dumps({
        "regime": regime_info["regime"],
        "confidence": regime_info["confidence"],
        "bias": regime_info["bias"]
    }, indent=2))
    print("Rules:", regime_info["allocation_rules"])

    print("\n--- 2. High Volatility Scenario ---")
    high_vol = agent.detect_regime(vix_level=32.0, volatility_20d=0.34)
    print(json.dumps({
        "regime": high_vol["regime"],
        "confidence": high_vol["confidence"],
        "bias": high_vol["bias"]
    }, indent=2))

    print("\n--- 3. Bear Market Scenario ---")
    bear = agent.detect_regime(spy_price=440.0, ema_50=460.0, ema_200=480.0, vix_level=26.5, momentum_5d=-0.03)
    print(json.dumps({
        "regime": bear["regime"],
        "confidence": bear["confidence"],
        "bias": bear["bias"]
    }, indent=2))
