import logging
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    confusion_matrix,
    classification_report,
)
from sklearn.calibration import calibration_curve

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ModelEvaluator")


def evaluate_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    returns_5d: Optional[np.ndarray] = None,
    risk_free_rate: float = 0.04,
) -> Dict[str, Any]:
    """
    Computes comprehensive classification, financial, and probability calibration metrics.
    """
    y_true_arr = np.asarray(y_true).astype(int)
    y_pred_arr = np.asarray(y_pred).astype(int)
    y_prob_arr = np.asarray(y_prob).astype(float)

    # 1. Classification Metrics
    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    prec = float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred_arr, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))
    try:
        auc = float(roc_auc_score(y_true_arr, y_prob_arr))
    except Exception:
        auc = 0.5

    # 2. Probability Calibration: Brier Score (lower is better, 0 is perfect)
    brier = float(brier_score_loss(y_true_arr, y_prob_arr))
    prob_true, prob_pred = calibration_curve(y_true_arr, y_prob_arr, n_bins=5, strategy="uniform")

    # 3. Financial Metrics on BUY Signals (where y_pred == 1)
    directional_accuracy = acc
    hit_rate = prec  # Precision on BUY signals: percentage of executed BUY signals that were profitable

    avg_return = 0.0
    sharpe_ratio = 0.0
    max_drawdown = 0.0

    if returns_5d is not None:
        ret_arr = np.asarray(returns_5d)
        buy_indices = np.where(y_pred_arr == 1)[0]
        if len(buy_indices) > 0:
            trade_returns = ret_arr[buy_indices]
            valid_returns = trade_returns[~np.isnan(trade_returns)]

            if len(valid_returns) > 0:
                avg_return = float(np.mean(valid_returns))
                std_ret = float(np.std(valid_returns))

                # Annualized Sharpe for 5-day holding periods (252 / 5 = 50.4 periods/yr)
                periods_per_year = 252 / 5.0
                rf_per_period = risk_free_rate / periods_per_year
                if std_ret > 1e-6:
                    sharpe_ratio = float((avg_return - rf_per_period) / std_ret * np.sqrt(periods_per_year))

                # Max Drawdown
                equity = np.cumprod(1 + valid_returns)
                peak = np.maximum.accumulate(equity)
                dd = (equity - peak) / peak
                max_drawdown = float(np.min(dd)) if len(dd) > 0 else 0.0

    cm = confusion_matrix(y_true_arr, y_pred_arr).tolist()

    return {
        "accuracy": round(acc, 4),
        "directional_accuracy": round(directional_accuracy, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(auc, 4),
        "brier_score": round(brier, 4),
        "hit_rate": round(hit_rate, 4),
        "avg_trade_return": round(avg_return * 100, 2),
        "sharpe_ratio": round(sharpe_ratio, 2),
        "max_drawdown": round(max_drawdown * 100, 2),
        "confusion_matrix": cm,
        "calibration": {
            "prob_true": [round(float(p), 4) for p in prob_true],
            "prob_pred": [round(float(p), 4) for p in prob_pred],
        },
    }


def format_comparison_table(results_dict: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Formats model evaluation metrics into a clean comparative leaderboard table."""
    rows = []
    for model_name, m in results_dict.items():
        rows.append({
            "Model": model_name,
            "Accuracy": f"{m['accuracy']:.2%}",
            "Directional Acc": f"{m['directional_accuracy']:.2%}",
            "ROC-AUC": f"{m['roc_auc']:.4f}",
            "F1-Score": f"{m['f1']:.4f}",
            "Hit Rate (Prec)": f"{m['hit_rate']:.2%}",
            "Avg Return": f"{m['avg_trade_return']:+.2f}%",
            "Sharpe": f"{m['sharpe_ratio']:.2f}",
            "Max DD": f"{m['max_drawdown']:.2f}%",
            "Brier Score": f"{m['brier_score']:.4f}",
        })
    df_comp = pd.DataFrame(rows)
    return df_comp
