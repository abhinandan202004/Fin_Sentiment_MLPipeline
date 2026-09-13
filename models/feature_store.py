import logging
from pathlib import Path
from typing import List, Tuple, Generator, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from config import DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("FeatureStore")

SENTIMENT_FEATURES = [
    "avg_sentiment",
    "sentiment_ema_3",
    "sentiment_ema_7",
    "sentiment_delta",
    "total_article_count",
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

ALL_FEATURES = SENTIMENT_FEATURES + TECHNICAL_FEATURES + REGIME_FEATURES
TARGET_COLUMN = "target"


def load_feature_dataset(
    csv_path: Optional[Path] = None,
    features: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Loads and cleans the training dataset from CSV or database.
    Ensures temporal sorting and removes trailing NaNs.
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
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values(by="date").reset_index(drop=True)

    selected_features = features if features is not None else ALL_FEATURES
    available_features = [f for f in selected_features if f in df.columns]

    # Clean numeric types
    for col in available_features:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    clean_df = df.dropna(subset=available_features + [TARGET_COLUMN]).copy()
    clean_df[TARGET_COLUMN] = clean_df[TARGET_COLUMN].astype(int)

    logger.info(
        f"Loaded {len(clean_df)} observations across {clean_df['ticker'].nunique()} tickers "
        f"({clean_df['date'].iloc[0]} to {clean_df['date'].iloc[-1]})."
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
