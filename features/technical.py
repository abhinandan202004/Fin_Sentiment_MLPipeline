import logging
from typing import Optional, Dict
import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
import yfinance as yf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TechnicalFeatures")


def compute_ticker_technical_indicators(df_ticker: pd.DataFrame) -> pd.DataFrame:
    """
    Computes technical indicators for a single ticker's OHLCV DataFrame sorted by date.
    Expects columns: ['date', 'open', 'high', 'low', 'close', 'volume'] or capitalized variants.
    """
    df = df_ticker.copy()
    # Normalize column names
    col_map = {c: c.capitalize() for c in df.columns if c.lower() in ["open", "high", "low", "close", "volume"]}
    df = df.rename(columns=col_map)

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # 1. Momentum: RSI(14)
    df["rsi"] = RSIIndicator(close=close, window=14).rsi()

    # 2. Trend: MACD(12, 26, 9)
    macd_ind = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd_ind.macd()
    df["macd_signal"] = macd_ind.macd_signal()
    df["macd_diff"] = macd_ind.macd_diff()

    # 3. Moving Averages: EMA20 & EMA50
    df["ema20"] = EMAIndicator(close=close, window=20).ema_indicator()
    df["ema50"] = EMAIndicator(close=close, window=50).ema_indicator()

    # 4. Volatility: Bollinger Bands (20, 2) & ATR (14)
    bb = BollingerBands(close=close, window=20, window_dev=2)
    df["bollinger_high"] = bb.bollinger_hband()
    df["bollinger_low"] = bb.bollinger_lband()
    df["bollinger_pband"] = bb.bollinger_pband()

    df["atr"] = AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()

    # 5. Price Momentum (Past Returns)
    df["return_1d"] = (close - close.shift(1)) / close.shift(1)
    df["return_5d"] = (close - close.shift(5)) / close.shift(5)
    df["return_20d"] = (close - close.shift(20)) / close.shift(20)

    # 6. Volume Ratio: volume / rolling_20d_avg_volume
    rolling_20d_vol = volume.rolling(window=20, min_periods=5).mean()
    df["volume_ratio"] = volume / (rolling_20d_vol + 1e-5)

    # 7. Labels: Forward Returns & Binary Target
    df["future_return_1d"] = (close.shift(-1) - close) / close
    df["future_return_5d"] = (close.shift(-5) - close) / close
    df["future_return_10d"] = (close.shift(-10) - close) / close
    df["target"] = (df["future_return_5d"] > 0).astype(int)

    return df


def fetch_spy_regime(period: str = "2y") -> pd.DataFrame:
    """Fetches SPY ETF to calculate market regime indicator: spy_return_5d."""
    logger.info("Fetching SPY benchmark for market regime feature...")
    try:
        spy = yf.Ticker("SPY")
        df_spy = spy.history(period=period, interval="1d")
        if df_spy.empty:
            return pd.DataFrame()
        df_spy = df_spy.dropna(subset=["Close"]).copy()
        df_spy["date"] = pd.to_datetime(df_spy.index).date
        df_spy["spy_return_5d"] = (df_spy["Close"] - df_spy["Close"].shift(5)) / df_spy["Close"].shift(5)
        return df_spy[["date", "spy_return_5d"]].dropna()
    except Exception as e:
        logger.warning(f"Failed to fetch SPY regime: {e}")
        return pd.DataFrame()


def compute_all_technical_features(
    df_market: pd.DataFrame,
    spy_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Computes technical indicators for each ticker in df_market and joins SPY 5-day market return.
    """
    if df_market.empty:
        return pd.DataFrame()

    results = []
    for ticker, group in df_market.groupby("ticker"):
        group_sorted = group.sort_values(by="date").copy()
        tech_group = compute_ticker_technical_indicators(group_sorted)
        results.append(tech_group)

    combined = pd.concat(results, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"]).dt.date

    # Join SPY regime feature
    if spy_df is not None and not spy_df.empty:
        spy_clean = spy_df.copy()
        spy_clean["date"] = pd.to_datetime(spy_clean["date"]).dt.date
        combined = pd.merge(combined, spy_clean[["date", "spy_return_5d"]], on="date", how="left")
    else:
        combined["spy_return_5d"] = 0.0

    # Fill any missing SPY returns with 0.0
    combined["spy_return_5d"] = combined["spy_return_5d"].fillna(0.0)

    return combined
