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
        csv_path = DATA_DIR / "training_dataset.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Training dataset not found at {csv_path}")

        df = pd.read_csv(csv_path)
        ticker_df = df[df["ticker"] == ticker].copy()
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

        # 1. RSI Condition
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

        # Synthesize technical narrative
        summary = (
            f"Ticker exhibits a {trend.replace('_', ' ').title()} with EMA20 above EMA50. "
            f"RSI sits at {rsi:.1f} ({rsi_cond}), indicating {rsi_cond.lower()} momentum. "
            f"MACD difference of {macd_diff:+.3f} reflects {macd_cond.replace('_', ' ').lower()}. "
            f"Volume ratio of {vol_ratio:.2f} indicates {vol_cond.lower()} institutional participation."
        )

        return {
            "ticker": ticker,
            "date": str(latest.get("date", "")),
            "rsi": round(rsi, 2),
            "rsi_condition": rsi_cond,
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
            "summary": summary,
        }


if __name__ == "__main__":
    ta = TechnicalAgent()
    res = ta.analyze_ticker("NVDA")
    print("\nTechnicalAgent Analysis for NVDA:")
    print(f"RSI: {res['rsi']} ({res['rsi_condition']})")
    print(f"MACD: {res['macd_condition']}")
    print(f"Trend: {res['trend']}")
    print(f"Summary: {res['summary']}")
