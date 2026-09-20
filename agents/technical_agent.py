import logging
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict, Any, Optional
import pandas as pd
from config import DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("TechnicalAgent")


class TechnicalAgent:
    """
    Responsible for:
    - RSI interpretation (momentum & overbought/oversold boundaries)
    - MACD trend & divergence analysis
    - Moving average structure (EMA20 vs EMA50)
    - Volume ratio expansion
    """
    def __init__(self):
        pass

    def analyze_ticker(self, ticker: str) -> Dict[str, Any]:
        from models.feature_store import load_feature_dataset
        df = load_feature_dataset()
        ticker_df = df[df["ticker"] == ticker.upper()].copy()
        if ticker_df.empty:
            raise ValueError(f"No market records for {ticker}")

        latest = ticker_df.sort_values(by="date").iloc[-1]

        rsi = float(latest.get("rsi", 50.0))
        macd = float(latest.get("macd", 0.0))
        macd_signal = float(latest.get("macd_signal", 0.0))
        macd_diff = float(latest.get("macd_diff", 0.0))
        ema20 = float(latest.get("ema20", 0.0))
        ema50 = float(latest.get("ema50", 0.0))
        vol_ratio = float(latest.get("volume_ratio", 1.0))
        atr = float(latest.get("atr", 0.0))
        bb_pband = float(latest.get("bollinger_pband", 0.5))

        # Sprint 4.1 Enhanced Quantitative Features
        vix_level = float(latest.get("vix_level", 20.0))
        sector_return_5d = float(latest.get("sector_return_5d", 0.0))
        rsi_rank = float(latest.get("rsi_rank", 0.50))
        bb_squeeze = float(latest.get("bb_squeeze", 0.0))
        spy_vol = float(latest.get("spy_volatility", 0.15))
        spy_5d = float(latest.get("spy_return_5d", 0.0))

        # Market Regime identification
        if vix_level > 25.0 or spy_vol > 0.22:
            market_regime = "High Volatility"
        elif vix_level < 15.0 and spy_vol < 0.14:
            market_regime = "Low Volatility"
        elif spy_5d > 0.01:
            market_regime = "Bull"
        elif spy_5d < -0.01:
            market_regime = "Bear"
        else:
            market_regime = "Neutral"

        # 1. RSI Condition & Cross-sectional Rank
        if rsi >= 75:
            rsi_cond = "OVERBOUGHT"
        elif rsi <= 30:
            rsi_cond = "OVERSOLD"
        elif rsi >= 55:
            rsi_cond = "MODERATELY_BULLISH"
        elif rsi <= 45:
            rsi_cond = "MODERATELY_BEARISH"
        else:
            rsi_cond = "NEUTRAL"

        # 2. MACD Condition
        if macd > macd_signal and macd_diff > 0:
            macd_cond = "BULLISH_CROSSOVER"
        elif macd < macd_signal and macd_diff < 0:
            macd_cond = "BEARISH_DIVERGENCE"
        else:
            macd_cond = "NEUTRAL"

        # 3. Moving Average Trend
        if ema20 > ema50:
            trend = "BULLISH_UPTREND"
        else:
            trend = "BEARISH_DOWNTREND"

        # 4. Volume Activity
        vol_cond = "HIGH_BREAKOUT" if vol_ratio > 1.3 else ("LOW" if vol_ratio < 0.7 else "NORMAL")

        # 5. Volatility Squeeze State
        squeeze_state = "SQUEEZE_ACTIVE" if bb_squeeze > 0.5 else "VOLATILITY_EXPANDING"

        # Synthesize enhanced technical narrative
        summary = (
            f"Technical setup indicates {trend.replace('_', ' ').title()} under a {market_regime} macro regime (VIX {vix_level:.1f}). "
            f"RSI sits at {rsi:.1f} (Rank {rsi_rank:.0%} cross-sectional, {rsi_cond}), "
            f"MACD difference of {macd_diff:+.3f} confirms {macd_cond.replace('_', ' ').lower()}. "
            f"Bollinger Squeeze status is {squeeze_state}, while 5-day sector momentum is {sector_return_5d:+.2%}."
        )

        return {
            "ticker": ticker.upper(),
            "date": str(latest.get("date", "")),
            "rsi": round(rsi, 2),
            "rsi_condition": rsi_cond,
            "rsi_rank": round(rsi_rank, 3),
            "macd": round(macd, 3),
            "macd_signal": round(macd_signal, 3),
            "macd_diff": round(macd_diff, 3),
            "macd_condition": macd_cond,
            "ema20": round(ema20, 2),
            "ema50": round(ema50, 2),
            "trend": trend,
            "volume_ratio": round(vol_ratio, 2),
            "volume_condition": vol_cond,
            "atr": round(atr, 2),
            "bollinger_pband": round(bb_pband, 3),
            "bb_squeeze": round(bb_squeeze, 2),
            "squeeze_state": squeeze_state,
            "vix_level": round(vix_level, 2),
            "market_regime": market_regime,
            "sector_return_5d": round(sector_return_5d, 4),
            "summary": summary,
        }


if __name__ == "__main__":
    ta = TechnicalAgent()
    for sym in ["NVDA", "AAPL", "JPM", "XOM"]:
        res = ta.analyze_ticker(sym)
        print(f"\nTechnicalAgent Analysis for {sym}:")
        print(f"Regime: {res['market_regime']} | VIX: {res['vix_level']} | Sector 5d: {res['sector_return_5d']:+.2%}")
        print(f"RSI: {res['rsi']} (Rank: {res['rsi_rank']:.0%}) | Squeeze: {res['squeeze_state']}")
        print(f"Summary: {res['summary']}")

