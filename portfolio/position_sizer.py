"""
Portfolio Position Sizer for Fin_Sentiment_MLPipeline.

Implements:
1. Kelly Criterion (Full & Half-Kelly)
2. Volatility Parity (1 / sigma weighting)
3. Fixed Fractional Risk Sizing
"""

import json
from typing import Dict, Any, Optional


class PositionSizer:
    """Calculates risk-adjusted position sizing using quantitative risk models."""

    def __init__(self, default_risk_fraction: float = 0.02, default_half_kelly: bool = True):
        self.default_risk_fraction = default_risk_fraction
        self.default_half_kelly = default_half_kelly

    def calculate_kelly(
        self,
        win_rate: float,
        reward_risk_ratio: float,
        half_kelly: bool = True,
        max_cap: float = 0.30
    ) -> float:
        """
        Calculates optimal position size using the Kelly Criterion:
        f* = p - (1 - p) / b
        where:
          p = probability of win (win rate)
          b = reward to risk ratio (win size / loss size)
        """
        if reward_risk_ratio <= 0:
            return 0.0

        p = min(max(win_rate, 0.0), 1.0)
        b = reward_risk_ratio

        f_star = p - ((1.0 - p) / b)
        
        # If negative edge, position size is 0
        if f_star <= 0:
            return 0.0

        fraction = f_star * 0.5 if half_kelly else f_star
        return round(min(fraction, max_cap), 4)

    def calculate_volatility_parity(
        self,
        asset_volatilities: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculates weights inversely proportional to asset volatility:
        w_i = (1 / sigma_i) / sum(1 / sigma_j)
        """
        inv_vols = {}
        for ticker, vol in asset_volatilities.items():
            if vol > 0.001:
                inv_vols[ticker] = 1.0 / vol
            else:
                inv_vols[ticker] = 1.0 / 0.001

        total_inv_vol = sum(inv_vols.values())
        if total_inv_vol == 0:
            n = len(asset_volatilities)
            return {t: round(1.0 / n, 4) for t in asset_volatilities}

        return {t: round(inv_vols[t] / total_inv_vol, 4) for t in asset_volatilities}

    def calculate_fixed_fractional(
        self,
        portfolio_value: float,
        entry_price: float,
        stop_loss_price: float,
        risk_fraction: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Calculates position size where max capital loss is limited to risk_fraction (e.g. 2%).
        """
        risk_pct = risk_fraction if risk_fraction is not None else self.default_risk_fraction
        risk_capital = portfolio_value * risk_pct

        per_share_risk = abs(entry_price - stop_loss_price)
        if per_share_risk <= 0 or entry_price <= 0:
            return {
                "shares": 0,
                "position_value": 0.0,
                "position_weight": 0.0,
                "risk_capital": risk_capital
            }

        shares = int(risk_capital / per_share_risk)
        position_value = round(shares * entry_price, 2)
        weight = round(position_value / portfolio_value, 4) if portfolio_value > 0 else 0.0

        return {
            "shares": shares,
            "position_value": position_value,
            "position_weight": weight,
            "risk_capital": round(risk_capital, 2)
        }

    def recommend_size(
        self,
        ticker: str,
        win_rate: float = 0.56,
        reward_risk_ratio: float = 1.35,
        volatility: float = 0.25,
        max_position: float = 0.30,
        regime: str = "Bull"
    ) -> Dict[str, Any]:
        """
        Generates a consensus recommended position size incorporating Kelly,
        volatility risk, and regime-aware scaling.
        """
        kelly_size = self.calculate_kelly(
            win_rate=win_rate,
            reward_risk_ratio=reward_risk_ratio,
            half_kelly=self.default_half_kelly,
            max_cap=max_position
        )

        # High Volatility regime cuts Kelly sizing by 50%
        if regime in ("High Volatility", "Bear"):
            kelly_size = round(kelly_size * 0.50, 4)

        # Cap based on single asset limit
        final_weight = min(kelly_size, max_position)

        return {
            "ticker": ticker,
            "kelly_half": kelly_size,
            "recommended_position": final_weight,
            "regime_adjustment": "50% cut" if regime in ("High Volatility", "Bear") else "normal"
        }


if __name__ == "__main__":
    sizer = PositionSizer()
    print("--- Testing Kelly Criterion ---")
    kelly = sizer.calculate_kelly(win_rate=0.60, reward_risk_ratio=1.5, half_kelly=True)
    print(f"Half Kelly (p=0.60, b=1.5): {kelly}")

    print("\n--- Testing Volatility Parity ---")
    vols = {"NVDA": 0.35, "AAPL": 0.20, "MSFT": 0.22}
    vp = sizer.calculate_volatility_parity(vols)
    print(f"Volatility Parity Weights: {vp}")

    print("\n--- Testing Fixed Fractional ---")
    ff = sizer.calculate_fixed_fractional(100000, entry_price=120.0, stop_loss_price=110.0)
    print(f"Fixed Fractional (100k, 2% risk): {ff}")

    print("\n--- Testing Consensus Recommendation ---")
    rec = sizer.recommend_size("NVDA", win_rate=0.62, reward_risk_ratio=1.4, volatility=0.32, regime="High Volatility")
    print(f"Consensus: {json.dumps(rec, indent=2)}")
