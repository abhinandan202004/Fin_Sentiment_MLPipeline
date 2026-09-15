import logging
from pathlib import Path
from typing import List, Tuple, Generator, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from config import DATA_DIR

# Sprint 7: Import engineered feature list (imported lazily inside functions
# to avoid circular imports during dataset build).

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("FeatureStore")

SENTIMENT_FEATURES = [
    "avg_sentiment",
    "sentiment_ema_3",
    "sentiment_ema_7",
    "sentiment_delta",
    "total_article_count",
    "positive_article_count",
    "negative_article_count",
]

TECHNICAL_FEATURES = [
    "rsi",
    "macd",
    "macd_signal",
    "macd_diff",
    "ema20",
    "ema50",
    "bollinger_pband",
    "atr",
    "return_1d",
    "return_5d",
    "return_20d",
    "volume_ratio",
]

REGIME_FEATURES = [
    "spy_return_5d",
    "spy_return_1d",
    "spy_volatility",
    "vix_level",
    "sector_return_5d",
]

# Sprint 7: Cross-sectional rank features
CROSS_SECTIONAL_FEATURES = [
    "rsi_rank",
    "sentiment_rank",
    "return_5d_rank",
    "volume_rank",
]

# Sprint 7: Interaction and composite features
INTERACTION_FEATURES = [
    "sentiment_x_momentum",
    "sentiment_x_volume",
    "sentiment_x_rsi_rank",
]

# Sprint 7: Additional regime and crossover signals
ENGINEERED_SIGNAL_FEATURES = [
    "vix_regime",
    "sector_alpha",
    "price_vs_52w_high",
    "bb_squeeze",
    "rsi_cross_50",
    "macd_positive_crossover",
    "volume_surge",
    "rsi_overbought",
    "rsi_oversold",
    "lag_return_1d",
    "lag_return_5d",
    "lag_sentiment",
]

ALL_FEATURES = (
    SENTIMENT_FEATURES
    + TECHNICAL_FEATURES
    + REGIME_FEATURES
    + CROSS_SECTIONAL_FEATURES
    + INTERACTION_FEATURES
    + ENGINEERED_SIGNAL_FEATURES
)
TARGET_COLUMN = "target"


def load_feature_dataset(
    csv_path: Optional[Path] = None,
    features: Optional[List[str]] = None,
    run_feature_engineering: bool = True,
) -> pd.DataFrame:
    """
    Loads and cleans the training dataset from CSV, optionally running
    the Sprint 7 feature engineering pipeline to add cross-sectional
    rank features, interaction terms, and regime signals.

    Parameters
    ----------
    csv_path : Path, optional
        Path to the training CSV (default: data/training_dataset.csv).
    features : list, optional
        Explicit feature list to load (default: ALL_FEATURES).
    run_feature_engineering : bool
        If True (default), calls feature_engineer.py to add engineered
        features before returning.  Set False if the CSV already
        contains the engineered columns.
    """
    if csv_path is None:
        csv_path = DATA_DIR / "training_dataset.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. Please run scripts/build_training_dataset.py first."
        )

    logger.info(f"Loading feature dataset from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Sort strictly by date to preserve temporal structure
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by=["date", "ticker"]).reset_index(drop=True)

    # Sprint 7: Run feature engineering to add cross-sectional & interaction features
    if run_feature_engineering:
        # Check if engineered features already present (avoid recomputing)
        from models.feature_engineer import engineer_features, ALL_ENGINEERED_FEATURES
        missing_engineered = [c for c in ALL_ENGINEERED_FEATURES if c not in df.columns]
        if missing_engineered:
            logger.info(
                f"Running feature engineering ({len(missing_engineered)} new features)..."
            )
            df, _ = engineer_features(df)
        else:
            logger.info("Engineered features already present in dataset — skipping recomputation.")

    # Normalise date back to date objects for compatibility
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values(by="date").reset_index(drop=True)

    selected_features = features if features is not None else ALL_FEATURES
    available_features = [f for f in selected_features if f in df.columns]

    # Clean numeric types
    for col in available_features:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    clean_df = df.dropna(subset=[TARGET_COLUMN]).copy()
    clean_df[TARGET_COLUMN] = clean_df[TARGET_COLUMN].astype(int)

    logger.info(
        f"Loaded {len(clean_df)} observations across {clean_df['ticker'].nunique()} tickers "
        f"({clean_df['date'].iloc[0]} to {clean_df['date'].iloc[-1]}) "
        f"with {len(available_features)} features."
    )
    return clean_df


def temporal_train_test_split(
    df: pd.DataFrame,
    split_ratio: float = 0.80,
    features: Optional[List[str]] = None,
    target_col: str = TARGET_COLUMN,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Strict temporal walk-forward split (no shuffling, no lookahead data leakage).
    """
    if features is None:
        features = [f for f in ALL_FEATURES if f in df.columns]
    else:
        features = list(dict.fromkeys(features))

    df_sorted = df.sort_values(by="date").reset_index(drop=True)

    split_idx = int(len(df_sorted) * split_ratio)
    if split_idx <= 0 or split_idx >= len(df_sorted):
        raise ValueError("Insufficient observations for temporal train/test split.")

    train_df = df_sorted.iloc[:split_idx].copy()
    test_df = df_sorted.iloc[split_idx:].copy()

    X_train = train_df[features]
    y_train = train_df[target_col].astype(int)
    X_test = test_df[features]
    y_test = test_df[target_col].astype(int)

    logger.info(
        f"Temporal Split: Train={len(X_train)} samples ({train_df['date'].iloc[0]} to {train_df['date'].iloc[-1]}), "
        f"Test={len(X_test)} samples ({test_df['date'].iloc[0]} to {test_df['date'].iloc[-1]})"
    )

    return X_train, y_train, X_test, y_test, train_df, test_df


def get_time_series_cv(
    X: pd.DataFrame,
    n_splits: int = 5,
) -> TimeSeriesSplit:
    """Returns TimeSeriesSplit cross-validator for temporal hyperparameter search."""
    return TimeSeriesSplit(n_splits=n_splits)
