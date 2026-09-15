"""
models/threshold_optimizer.py

Sprint 7: Prediction Threshold Optimization
=============================================
Sweeps the prediction threshold (probability for classifiers, return
threshold for regressors) on the validation set to find the value that
maximises the target metric (Sharpe Ratio by default).

Background
----------
The default threshold of 0.50 is almost never optimal for financial
trading systems.  A model with 55% AUC may produce much better
risk-adjusted returns when only acting on high-confidence signals:

    Conservative: BUY if P(UP) > 0.62   → fewer trades, higher precision
    Aggressive:   BUY if P(UP) > 0.50   → more trades, lower precision

This module systematically answers: "What threshold maximises Sharpe?"
and persists the optimal threshold alongside the model artifact so that
the inference pipeline always uses the correct cut-off.

Usage
-----
    # From CLI
    python -m models.threshold_optimizer

    # Programmatically
    from models.threshold_optimizer import optimize_threshold, plot_threshold_curve
    result = optimize_threshold(y_val, proba_val, returns_val)
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import ARTIFACTS_DIR, DATA_DIR
from models.feature_store import load_feature_dataset, temporal_train_test_split, ALL_FEATURES
from models.feature_engineer import engineer_features, ALL_ENGINEERED_FEATURES
from models.target_engineering import evaluate_regression_direction

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("ThresholdOptimizer")


# ---------------------------------------------------------------------------
# Core threshold sweep
# ---------------------------------------------------------------------------

def optimize_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    returns_true: Optional[np.ndarray] = None,
    metric: str = "sharpe",
    thresholds: Optional[np.ndarray] = None,
    is_regression: bool = False,
) -> Dict[str, Any]:
    """
    Sweeps prediction thresholds and finds the optimal cut-off.

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground-truth labels (0/1) or ignored when is_regression=True.
    y_score : np.ndarray
        Predicted probability (classifier) or predicted return (regressor).
    returns_true : np.ndarray, optional
        Actual 5-day returns for Sharpe calculation.  Required when
        metric in ('sharpe', 'expected_return').
    metric : str
        Optimisation objective. Options:
          'sharpe'          – Annualised Sharpe on BUY-signal trades (recommended)
          'precision'       – Fraction of BUY signals that were profitable
          'expected_return' – Mean return per triggered BUY trade
          'f1'              – F1 score on the BUY class
          'accuracy'        – Overall directional accuracy
    thresholds : np.ndarray, optional
        Candidate thresholds to sweep (default: np.arange(0.45, 0.80, 0.01)).
    is_regression : bool
        If True, y_score contains predicted returns (not probabilities).

    Returns
    -------
    dict with keys: optimal_threshold, metric_name, metric_value,
                    n_signals_at_optimal, full_curve (DataFrame)
    """
    if thresholds is None:
        if is_regression:
            # For regression, sweep return thresholds from -1% to +3%
            thresholds = np.arange(-0.010, 0.031, 0.002)
        else:
            thresholds = np.arange(0.45, 0.80, 0.01)

    records = []
    for t in thresholds:
        signals = (y_score >= t).astype(int)
        n_signals = signals.sum()

        if n_signals == 0:
            records.append({
                "threshold": round(float(t), 4),
                "n_signals": 0,
                "signal_rate": 0.0,
                "accuracy": 0.0,
                "precision": 0.0,
                "sharpe": -99.0,
                "expected_return": 0.0,
                "f1": 0.0,
            })
            continue

        # --- Classification metrics ---
        if not is_regression:
            actual_dir = y_true
        else:
            actual_dir = (returns_true > 0).astype(int) if returns_true is not None else None

        accuracy = (signals == actual_dir).mean() if actual_dir is not None else 0.0

        # Precision on BUY signals
        if n_signals > 0 and actual_dir is not None:
            precision = actual_dir[signals == 1].mean()
        else:
            precision = 0.0

        # F1 score
        if actual_dir is not None:
            tp = ((signals == 1) & (actual_dir == 1)).sum()
            fp = ((signals == 1) & (actual_dir == 0)).sum()
            fn = ((signals == 0) & (actual_dir == 1)).sum()
            f1 = (2 * tp) / (2 * tp + fp + fn + 1e-9)
        else:
            f1 = 0.0

        # --- Financial metrics (require actual returns) ---
        if returns_true is not None and n_signals > 0:
            buy_returns = returns_true[signals == 1]
            expected_return = buy_returns.mean() * 100
            if len(buy_returns) > 1 and buy_returns.std() > 0:
                sharpe = (buy_returns.mean() / buy_returns.std()) * np.sqrt(252 / 5)
            else:
                sharpe = 0.0
        else:
            expected_return = 0.0
            sharpe = 0.0

        records.append({
            "threshold": round(float(t), 4),
            "n_signals": int(n_signals),
            "signal_rate": round(float(n_signals / len(signals)), 4),
            "accuracy": round(float(accuracy), 4),
            "precision": round(float(precision), 4),
            "sharpe": round(float(sharpe), 4),
            "expected_return": round(float(expected_return), 4),
            "f1": round(float(f1), 4),
        })

    curve_df = pd.DataFrame(records)

    # Find optimal threshold
    metric_col_map = {
        "sharpe":          "sharpe",
        "precision":       "precision",
        "expected_return": "expected_return",
        "f1":              "f1",
        "accuracy":        "accuracy",
    }
    sort_col = metric_col_map.get(metric, "sharpe")

    # Filter out thresholds with fewer than 5 signals (statistically unreliable)
    valid = curve_df[curve_df["n_signals"] >= 5]
    if valid.empty:
        valid = curve_df

    best_row = valid.loc[valid[sort_col].idxmax()]

    result = {
        "optimal_threshold":     float(best_row["threshold"]),
        "metric_name":           sort_col,
        "metric_value":          float(best_row[sort_col]),
        "n_signals_at_optimal":  int(best_row["n_signals"]),
        "signal_rate":           float(best_row["signal_rate"]),
        "precision_at_optimal":  float(best_row["precision"]),
        "sharpe_at_optimal":     float(best_row["sharpe"]),
        "full_curve":            curve_df,
    }

    logger.info(
        f"Optimal threshold: {result['optimal_threshold']:.4f} | "
        f"{sort_col}={result['metric_value']:.4f} | "
        f"signals={result['n_signals_at_optimal']} ({result['signal_rate']:.1%} of bars)"
    )
    return result


def print_threshold_curve(result: Dict[str, Any], top_n: int = 15) -> None:
    """Prints a formatted threshold curve table to stdout."""
    curve = result["full_curve"].copy()
    opt_t = result["optimal_threshold"]

    print("\n" + "=" * 80)
    print(" THRESHOLD OPTIMISATION CURVE")
    print(f" Optimising for: {result['metric_name'].upper()}")
    print("=" * 80)
    print(
        f"{'Threshold':>10} {'#Signals':>9} {'Signal%':>8} "
        f"{'Accuracy':>9} {'Precision':>10} {'Sharpe':>8} {'ExpRet%':>8}"
    )
    print("-" * 80)

    # Show subset around optimal
    subset = curve[curve["n_signals"] >= 3].copy()
    # Take top_n rows sorted by the metric
    metric = result["metric_name"]
    subset_sorted = subset.nlargest(top_n, metric)
    all_thresholds = pd.concat([subset_sorted, subset[subset["threshold"] == opt_t]]).drop_duplicates()
    all_thresholds = all_thresholds.sort_values("threshold")

    for _, row in all_thresholds.iterrows():
        marker = " <-- OPTIMAL" if abs(row["threshold"] - opt_t) < 1e-6 else ""
        print(
            f"{row['threshold']:>10.4f} {row['n_signals']:>9} {row['signal_rate']:>8.1%} "
            f"{row['accuracy']:>9.2%} {row['precision']:>10.2%} "
            f"{row['sharpe']:>8.3f} {row['expected_return']:>8.3f}%{marker}"
        )
    print("=" * 80)
    print(f" --> Optimal Threshold: {opt_t:.4f} | {metric}={result['metric_value']:.4f}")
    print(f" --> Signals generated: {result['n_signals_at_optimal']} "
          f"({result['signal_rate']:.1%} of test bars)")
    print("=" * 80 + "\n")


def save_optimal_threshold(threshold: float, metadata: Dict[str, Any]) -> None:
    """Persists the optimal threshold alongside model artifacts."""
    threshold_data = {"optimal_threshold": threshold, **metadata}
    path = ARTIFACTS_DIR / "optimal_threshold.json"
    with open(path, "w") as f:
        json.dump(threshold_data, f, indent=2)
    logger.info(f"Optimal threshold {threshold:.4f} saved to {path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """
    Loads the saved champion model, runs threshold sweep on held-out
    validation data, prints the curve, and persists the optimal threshold.
    """
    model_path = ARTIFACTS_DIR / "best_model.pkl"
    if not model_path.exists():
        logger.error("No trained model found.  Run models/train.py first.")
        sys.exit(1)

    model = joblib.load(model_path)
    logger.info(f"Loaded champion model from {model_path}")

    # Load dataset with engineered features (auto-engineered by feature_store)
    df = load_feature_dataset()
    fpath = ARTIFACTS_DIR / "feature_list.json"
    if fpath.exists():
        import json
        with open(fpath) as f:
            available_features = json.load(f)
    else:
        available_features = list(dict.fromkeys([f for f in ALL_FEATURES if f in df.columns]))

    X_train, y_train, X_val, y_val, train_df, val_df = temporal_train_test_split(
        df, split_ratio=0.80, features=available_features
    )
    returns_val = val_df["future_return_5d"].values

    # Get predicted probabilities
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_val)[:, 1]
        is_regression = False
        logger.info("Classifier detected — sweeping probability thresholds.")
    else:
        y_score = model.predict(X_val)
        is_regression = True
        logger.info("Regressor detected — sweeping return thresholds.")

    # --- Multi-metric sweep ---
    metrics_to_test = ["sharpe", "precision", "f1"]
    best_overall = None

    for metric in metrics_to_test:
        result = optimize_threshold(
            y_true=y_val.values,
            y_score=y_score,
            returns_true=returns_val,
            metric=metric,
            is_regression=is_regression,
        )
        print_threshold_curve(result, top_n=12)

        if metric == "sharpe":  # Primary metric
            best_overall = result

    # Persist optimal threshold (Sharpe-optimised)
    save_optimal_threshold(
        threshold=best_overall["optimal_threshold"],
        metadata={
            "metric": "sharpe",
            "sharpe": best_overall["sharpe_at_optimal"],
            "precision": best_overall["precision_at_optimal"],
            "n_signals": best_overall["n_signals_at_optimal"],
            "signal_rate": best_overall["signal_rate"],
        }
    )


if __name__ == "__main__":
    main()
