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


def fetch_market_regime(period: str = "2y") -> pd.DataFrame:
    """
    Fetches S&P 500 benchmark (SPY) and CBOE Volatility Index (^VIX)
    to calculate market regime features:
      - spy_return_1d
      - spy_return_5d
      - spy_volatility (20-day annualized rolling std)
      - vix_level
    """
    logger.info("Fetching SPY and ^VIX for market regime features...")
    try:
        spy = yf.Ticker("SPY").history(period=period, interval="1d")
        vix = yf.Ticker("^VIX").history(period=period, interval="1d")

        if spy.empty:
            return pd.DataFrame()

        spy = spy.dropna(subset=["Close"]).copy()
        spy_close = spy["Close"]
        spy["date"] = pd.to_datetime(spy.index).date
        spy["spy_return_1d"] = (spy_close - spy_close.shift(1)) / spy_close.shift(1)
        spy["spy_return_5d"] = (spy_close - spy_close.shift(5)) / spy_close.shift(5)
        log_ret = np.log(spy_close / spy_close.shift(1))
        spy["spy_volatility"] = log_ret.rolling(20, min_periods=5).std() * np.sqrt(252)

        regime_df = spy[["date", "spy_return_1d", "spy_return_5d", "spy_volatility"]].copy()

        if not vix.empty:
            vix = vix.dropna(subset=["Close"]).copy()
            vix["date"] = pd.to_datetime(vix.index).date
            vix["vix_level"] = vix["Close"]
            regime_df = pd.merge(regime_df, vix[["date", "vix_level"]], on="date", how="left")
            regime_df["vix_level"] = regime_df["vix_level"].ffill().bfill().fillna(20.0)
        else:
            regime_df["vix_level"] = 20.0

        return regime_df.dropna(subset=["spy_return_5d"])
    except Exception as e:
        logger.warning(f"Failed to fetch market regime: {e}")
        return pd.DataFrame()


# Alias for backward compatibility
fetch_spy_regime = fetch_market_regime


def compute_all_technical_features(
    df_market: pd.DataFrame,
    regime_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Computes technical indicators for each ticker in df_market and joins market regime features.
    Also computes sector_return_5d as cross-sectional average return across tickers in same sector.
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

    # Join Market Regime features (SPY returns, SPY volatility, VIX)
    if regime_df is not None and not regime_df.empty:
        reg_clean = regime_df.copy()
        reg_clean["date"] = pd.to_datetime(reg_clean["date"]).dt.date
        cols_to_join = ["date", "spy_return_1d", "spy_return_5d", "spy_volatility", "vix_level"]
        available_cols = [c for c in cols_to_join if c in reg_clean.columns]
        combined = pd.merge(combined, reg_clean[available_cols], on="date", how="left")
    else:
        combined["spy_return_1d"] = 0.0
        combined["spy_return_5d"] = 0.0
        combined["spy_volatility"] = 0.15
        combined["vix_level"] = 20.0

    # Fill any missing regime values with reasonable baseline defaults
    combined["spy_return_1d"] = combined["spy_return_1d"].fillna(0.0)
    combined["spy_return_5d"] = combined["spy_return_5d"].fillna(0.0)
    combined["spy_volatility"] = combined["spy_volatility"].fillna(0.15)
    combined["vix_level"] = combined["vix_level"].fillna(20.0)

    # Compute sector_return_5d: cross-sectional mean of 5-day return across tickers per date
    daily_cross_sec = combined.groupby("date")["return_5d"].mean().reset_index()
    daily_cross_sec.rename(columns={"return_5d": "sector_return_5d"}, inplace=True)
    combined = pd.merge(combined, daily_cross_sec, on="date", how="left")
    combined["sector_return_5d"] = combined["sector_return_5d"].fillna(0.0)

    return combined
