"""
models/feature_engineer.py

Sprint 7: Advanced Feature Engineering Layer
==========================================
Computes three categories of high-signal features AFTER the base
technical/sentiment pipeline has built the raw training DataFrame:

1. Cross-Sectional Rank Features
   - RSI, sentiment, 5d-return, and volume-ratio percentile ranks
     computed per date across all tickers in the universe.
   - Rationale: "RSI=65" is ambiguous; "RSI=65 is top-80th percentile
     today" is actionable. Cross-sectional signals are far more
     predictive than absolute values for a relative-value strategy.

2. Interaction & Composite Features
   - sentiment × momentum, sentiment × volume, sentiment × RSI_rank.
   - RSI/MACD crossover binary flags, Bollinger Band squeeze.
   - Price relative to 52-week high (proxy for breakout strength).
   - Sector alpha (ticker return minus SPY return).

3. Lagged Returns (Momentum Persistence)
   - Prev-day and prev-week returns to capture short-term momentum
     vs mean-reversion patterns.

All features are added in-place to the input DataFrame and a complete
list of NEW column names is returned for use by the feature store.
"""

import logging
from typing import List, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger("FeatureEngineer")

# ---------------------------------------------------------------------------
# Column groups produced by this module
# ---------------------------------------------------------------------------
CROSS_SECTIONAL_FEATURES: List[str] = [
    "rsi_rank",
    "sentiment_rank",
    "return_5d_rank",
    "volume_rank",
]

INTERACTION_FEATURES: List[str] = [
    "sentiment_x_momentum",
    "sentiment_x_volume",
    "sentiment_x_rsi_rank",
]

REGIME_SIGNAL_FEATURES: List[str] = [
    "vix_regime",           # 1 = High Fear (VIX>20), 0 = Low Fear
    "sector_alpha",         # ticker 5d return minus SPY 5d return
    "price_vs_52w_high",    # close / 252-day rolling max
    "bb_squeeze",           # Bollinger Band width percentile (low = squeeze)
]

CROSSOVER_FEATURES: List[str] = [
    "rsi_cross_50",              # RSI crossed above 50 from below
    "macd_positive_crossover",   # MACD line crossed above signal line
    "volume_surge",              # volume_ratio > 1.5
    "rsi_overbought",            # rsi > 70
    "rsi_oversold",              # rsi < 30
]

LAG_FEATURES: List[str] = [
    "lag_return_1d",
    "lag_return_5d",
    "lag_sentiment",
]

ALL_ENGINEERED_FEATURES: List[str] = (
    CROSS_SECTIONAL_FEATURES
    + INTERACTION_FEATURES
    + REGIME_SIGNAL_FEATURES
    + CROSSOVER_FEATURES
    + LAG_FEATURES
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Main entry point. Adds all engineered features to ``df`` and returns
    (enriched_df, list_of_new_column_names).

    Parameters
    ----------
    df : pd.DataFrame
        Must already contain the base columns produced by the technical and
        sentiment pipelines (rsi, macd, macd_signal, sentiment features,
        return_*d, volume_ratio, vix_level, spy_return_5d, bollinger_high,
        bollinger_low, close).

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        (DataFrame with new columns appended, list of new column names)
    """
    df = df.copy()

    already_engineered = [f for f in ALL_ENGINEERED_FEATURES if f in df.columns]
    if len(already_engineered) == len(ALL_ENGINEERED_FEATURES):
        logger.info("Engineered features already present in DataFrame. Skipping duplicate engineering.")
        return df, already_engineered

    # Sort strictly by date (temporal integrity)
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    logger.info("Engineering cross-sectional rank features...")
    df = _add_cross_sectional_ranks(df)

    logger.info("Engineering interaction features...")
    df = _add_interaction_features(df)

    logger.info("Engineering regime signal features...")
    df = _add_regime_signals(df)

    logger.info("Engineering crossover / flag features...")
    df = _add_crossover_flags(df)

    logger.info("Engineering lag features...")
    df = _add_lag_features(df)

    # Verify no inadvertent NaN inflation in engineered columns
    existing_new_cols = [c for c in ALL_ENGINEERED_FEATURES if c in df.columns]
    nan_report = df[existing_new_cols].isna().mean()
    high_nan = nan_report[nan_report > 0.10]
    if not high_nan.empty:
        logger.warning(f"Engineered columns with >10% NaN:\n{high_nan}")

    # Fill remaining NaNs with 0 for non-rank features
    for col in existing_new_cols:
        if col.endswith("_rank"):
            df[col] = df[col].fillna(0.5)  # neutral rank
        else:
            df[col] = df[col].fillna(0.0)

    logger.info(
        f"Feature engineering complete. Added {len(existing_new_cols)} new columns: "
        f"{existing_new_cols}"
    )
    return df, existing_new_cols


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _add_cross_sectional_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-date percentile rank of RSI, sentiment, 5-day return, and volume ratio
    across all tickers in the universe.  Rank in [0, 1] (higher = stronger).
    """
    rank_map = {
        "rsi_rank":       "rsi",
        "sentiment_rank": "avg_sentiment",
        "return_5d_rank": "return_5d",
        "volume_rank":    "volume_ratio",
    }
    for new_col, src_col in rank_map.items():
        if src_col not in df.columns:
            logger.warning(f"Source column '{src_col}' missing; skipping {new_col}.")
            df[new_col] = 0.5
            continue
        df[new_col] = (
            df.groupby("date")[src_col]
            .rank(pct=True, na_option="keep")
        )
    return df


def _add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Multiplicative interaction terms that capture joint signals.
    """
    # sentiment × 5-day momentum
    if "avg_sentiment" in df.columns and "return_5d" in df.columns:
        df["sentiment_x_momentum"] = df["avg_sentiment"] * df["return_5d"]
    else:
        df["sentiment_x_momentum"] = 0.0

    # sentiment × volume burst
    if "avg_sentiment" in df.columns and "volume_ratio" in df.columns:
        df["sentiment_x_volume"] = df["avg_sentiment"] * df["volume_ratio"]
    else:
        df["sentiment_x_volume"] = 0.0

    # sentiment × RSI cross-sectional rank (high sentiment + top-ranked RSI = very bullish)
    if "avg_sentiment" in df.columns and "rsi_rank" in df.columns:
        df["sentiment_x_rsi_rank"] = df["avg_sentiment"] * df["rsi_rank"]
    else:
        df["sentiment_x_rsi_rank"] = 0.0

    return df


def _add_regime_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Market-regime and relative-price features.
    """
    # VIX fear regime (binary)
    if "vix_level" in df.columns:
        df["vix_regime"] = (df["vix_level"] > 20).astype(float)
    else:
        df["vix_regime"] = 0.0

    # Sector alpha = how much is this ticker outperforming SPY?
    if "return_5d" in df.columns and "spy_return_5d" in df.columns:
        df["sector_alpha"] = df["return_5d"] - df["spy_return_5d"]
    else:
        df["sector_alpha"] = 0.0

    # Price vs 252-day rolling high (breakout proximity)
    if "close" in df.columns:
        df["price_vs_52w_high"] = (
            df.groupby("ticker")["close"]
            .transform(lambda x: x / x.rolling(252, min_periods=20).max())
        )
    else:
        df["price_vs_52w_high"] = 1.0

    # Bollinger Band squeeze: width percentile over past 252 days (low = compression)
    if "bollinger_high" in df.columns and "bollinger_low" in df.columns:
        bb_width = df["bollinger_high"] - df["bollinger_low"]
        df["bb_squeeze"] = (
            df.groupby("ticker")["bb_width"]
            .transform(lambda x: x.rank(pct=True))
            if "bb_width" in df.columns
            else bb_width.groupby(df["ticker"]).transform(
                lambda x: x.rank(pct=True)
            )
        )
        # Reassign cleanly
        df["_bb_width"] = bb_width
        df["bb_squeeze"] = df.groupby("ticker")["_bb_width"].transform(
            lambda x: x.rank(pct=True, na_option="keep")
        )
        df.drop(columns=["_bb_width"], inplace=True, errors="ignore")
    else:
        df["bb_squeeze"] = 0.5

    return df


def _add_crossover_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Binary flag features for momentum crossover events.
    """
    if "rsi" in df.columns:
        rsi_prev = df.groupby("ticker")["rsi"].shift(1)
        df["rsi_cross_50"] = (
            (df["rsi"] >= 50) & (rsi_prev < 50)
        ).astype(float)
        df["rsi_overbought"] = (df["rsi"] > 70).astype(float)
        df["rsi_oversold"]   = (df["rsi"] < 30).astype(float)
    else:
        df["rsi_cross_50"]   = 0.0
        df["rsi_overbought"] = 0.0
        df["rsi_oversold"]   = 0.0

    if "macd" in df.columns and "macd_signal" in df.columns:
        macd_prev = df.groupby("ticker")["macd"].shift(1)
        signal_prev = df.groupby("ticker")["macd_signal"].shift(1)
        df["macd_positive_crossover"] = (
            (df["macd"] > df["macd_signal"]) & (macd_prev <= signal_prev)
        ).astype(float)
    else:
        df["macd_positive_crossover"] = 0.0

    if "volume_ratio" in df.columns:
        df["volume_surge"] = (df["volume_ratio"] > 1.5).astype(float)
    else:
        df["volume_surge"] = 0.0

    return df


def _add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lagged features to capture momentum persistence / mean-reversion.
    """
    if "return_1d" in df.columns:
        df["lag_return_1d"] = df.groupby("ticker")["return_1d"].shift(1)
    else:
        df["lag_return_1d"] = 0.0

    if "return_5d" in df.columns:
        df["lag_return_5d"] = df.groupby("ticker")["return_5d"].shift(1)
    else:
        df["lag_return_5d"] = 0.0

    if "avg_sentiment" in df.columns:
        df["lag_sentiment"] = df.groupby("ticker")["avg_sentiment"].shift(1)
    else:
        df["lag_sentiment"] = 0.0

    return df
