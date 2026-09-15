"""
models/target_engineering.py

Sprint 7: Advanced Target Engineering
======================================
Provides two complementary prediction targets that replace the naive
`future_return_5d > 0` binary label:

1. Regression Target
   - Raw `future_return_5d` (float).
   - Train XGBoost/LightGBM as regressors.
   - Convert to directional signal post-hoc: BUY if predicted_return > threshold.
   - No information is thrown away; the model learns the magnitude of moves.

2. Three-Class Classification Target
   - UP:      future_return_5d >  conviction_threshold  (e.g. +1.5%)
   - NEUTRAL: |future_return_5d| <= conviction_threshold
   - DOWN:    future_return_5d < -conviction_threshold
   - Preserves ALL training rows (unlike the "drop neutral" approach).
   - Neutral class still provides valuable signal about market equilibrium.
   - For the binary BUY/SELL decision, collapse UP vs (NEUTRAL+DOWN).

Both modes are supported and can be combined in the training pipeline.
"""

import logging
from typing import Tuple, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger("TargetEngineering")

# Default conviction thresholds (calibrated to large-cap stock volatility)
DEFAULT_CONVICTION_THRESHOLD = 0.015   # ±1.5% 5-day move
STRONG_CONVICTION_THRESHOLD  = 0.025   # ±2.5% — higher precision, fewer signals


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_regression_target(df: pd.DataFrame, col: str = "future_return_5d") -> pd.Series:
    """
    Returns the raw float future return as the regression target.

    This preserves full information about return magnitude.  The model
    learns to rank stocks by expected return, not just binary direction.

    Parameters
    ----------
    df : pd.DataFrame
    col : str  Column name of the forward return to use as target.

    Returns
    -------
    pd.Series  Float target (same index as df).
    """
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found.  Available: {list(df.columns)}")
    y = pd.to_numeric(df[col], errors="coerce")
    n_null = y.isna().sum()
    if n_null > 0:
        logger.warning(f"Regression target has {n_null} NaN rows; these will be dropped.")
    logger.info(
        f"Regression target '{col}': mean={y.mean():.4f}, std={y.std():.4f}, "
        f"min={y.min():.4f}, max={y.max():.4f}"
    )
    return y


def build_3class_target(
    df: pd.DataFrame,
    col: str = "future_return_5d",
    threshold: float = DEFAULT_CONVICTION_THRESHOLD,
) -> Tuple[pd.Series, dict]:
    """
    Creates a three-class ordinal target preserving all rows.

    Classes:
        2 = UP      (future_return_5d >  +threshold)
        1 = NEUTRAL (|future_return_5d| <= threshold)
        0 = DOWN    (future_return_5d < -threshold)

    Parameters
    ----------
    df : pd.DataFrame
    col : str  Column with the forward return value.
    threshold : float  Conviction boundary (default ±1.5%).

    Returns
    -------
    Tuple[pd.Series, dict]
        (3-class label series, class distribution stats dict)
    """
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found.")

    ret = pd.to_numeric(df[col], errors="coerce")
    labels = pd.Series(1, index=df.index, dtype=int, name="target_3class")
    labels[ret >  threshold] = 2   # UP
    labels[ret < -threshold] = 0   # DOWN

    dist = labels.value_counts(normalize=True).round(3).to_dict()
    stats = {
        "threshold": threshold,
        "UP_pct":      dist.get(2, 0.0),
        "NEUTRAL_pct": dist.get(1, 0.0),
        "DOWN_pct":    dist.get(0, 0.0),
        "total_rows":  len(labels),
    }
    logger.info(
        f"3-class target (±{threshold:.1%}) | "
        f"UP={stats['UP_pct']:.1%}, NEUTRAL={stats['NEUTRAL_pct']:.1%}, "
        f"DOWN={stats['DOWN_pct']:.1%} | total={stats['total_rows']:,}"
    )
    return labels, stats


def build_binary_from_regression(
    predicted_returns: np.ndarray,
    threshold: float = 0.005,
) -> np.ndarray:
    """
    Converts regressor predictions (expected returns) to binary BUY/SELL signals.

    Instead of the default 0.5 probability cutoff (which is arbitrary),
    this lets you apply an asymmetric return threshold:

        BUY  if predicted_return >  threshold
        SELL if predicted_return <= threshold

    Tuning `threshold` on validation data directly optimizes Sharpe ratio.

    Parameters
    ----------
    predicted_returns : np.ndarray  Model's predicted 5-day return.
    threshold : float  Minimum expected return to trigger BUY (default 0.5%).

    Returns
    -------
    np.ndarray  Binary array (1 = BUY, 0 = SELL/HOLD).
    """
    return (predicted_returns > threshold).astype(int)


def add_conviction_weight(
    df: pd.DataFrame,
    col: str = "future_return_5d",
    threshold: float = DEFAULT_CONVICTION_THRESHOLD,
) -> pd.Series:
    """
    Optional: sample weights for training that down-weight near-zero return rows.

    Observations with |return| near zero are noisy labels (random walk territory).
    Weighting them less tells the model to focus on learning from clear moves.

    Weight formula:
        weight = min(|return| / threshold, 1.0) + 0.3
        (0.3 floor ensures neutral rows still contribute; capped at 1.3 for extremes)

    Parameters
    ----------
    df : pd.DataFrame
    col : str  Column with the forward return value.
    threshold : float  Conviction boundary for full weight (default ±1.5%).

    Returns
    -------
    pd.Series  Sample weights in [0.3, 1.3].
    """
    ret = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    weight = (ret.abs() / threshold).clip(upper=1.0) + 0.3
    logger.info(
        f"Conviction weights: mean={weight.mean():.3f}, "
        f"min={weight.min():.3f}, max={weight.max():.3f}"
    )
    return weight


def evaluate_regression_direction(
    y_true_returns: np.ndarray,
    y_pred_returns: np.ndarray,
    threshold: float = 0.005,
) -> dict:
    """
    Evaluates a regression model on directional accuracy and financial metrics.
    This is the key bridge between regression output and trading performance.

    Parameters
    ----------
    y_true_returns : np.ndarray  Actual 5-day forward returns.
    y_pred_returns : np.ndarray  Predicted 5-day forward returns.
    threshold : float  BUY signal threshold applied to predicted returns.

    Returns
    -------
    dict  Directional accuracy, hit rate, avg return, Sharpe, max drawdown.
    """
    signals = build_binary_from_regression(y_pred_returns, threshold=threshold)
    actual_direction = (y_true_returns > 0).astype(int)

    # Directional accuracy (did we call the direction correctly?)
    dir_acc = (signals == actual_direction).mean()

    # Hit rate on BUY signals (precision for class=1)
    buy_mask = signals == 1
    if buy_mask.sum() == 0:
        hit_rate = 0.0
        avg_return = 0.0
    else:
        hit_rate  = (y_true_returns[buy_mask] > 0).mean()
        avg_return = y_true_returns[buy_mask].mean() * 100

    # Sharpe on BUY signals
    if buy_mask.sum() > 1:
        buy_returns = y_true_returns[buy_mask]
        sharpe = (buy_returns.mean() / (buy_returns.std() + 1e-9)) * np.sqrt(252 / 5)
    else:
        sharpe = 0.0

    # Max drawdown on cumulative BUY-signal returns
    cum_ret = np.cumprod(1 + y_true_returns * signals) - 1
    rolling_max = np.maximum.accumulate(cum_ret + 1)
    drawdowns = (cum_ret + 1) / rolling_max - 1
    max_dd = drawdowns.min() * 100

    return {
        "directional_accuracy": round(float(dir_acc), 4),
        "hit_rate":             round(float(hit_rate), 4),
        "avg_trade_return_pct": round(float(avg_return), 4),
        "sharpe_ratio":         round(float(sharpe), 4),
        "max_drawdown_pct":     round(float(max_dd), 4),
        "n_buy_signals":        int(buy_mask.sum()),
        "threshold_used":       threshold,
    }
