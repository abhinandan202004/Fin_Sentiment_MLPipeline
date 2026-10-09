"""
Evaluation Metrics Engine for Fin_Sentiment_MLPipeline.
Sprint 9: Autonomous Evaluation & Adaptive Intelligence.

Contains both Statistical and Financial evaluation metrics to distinguish
between pure classification accuracy and genuine out-of-sample economic value.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    log_loss
)


# =====================================================================
# 1. STATISTICAL METRICS
# =====================================================================

def compute_accuracy(y_true: Union[List, np.ndarray], y_pred: Union[List, np.ndarray]) -> float:
    """Computes binary classification accuracy."""
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    if len(y_t) == 0:
        return 0.0
    return float(accuracy_score(y_t, y_p))


def compute_directional_accuracy(returns_or_y_true: Union[List, np.ndarray],
                                 signals_or_y_pred: Union[List, np.ndarray]) -> float:
    """
    Computes directional accuracy:
    Fraction of times sign of prediction matches sign of realized return.
    """
    y_t = np.asarray(returns_or_y_true, dtype=float)
    y_p = np.asarray(signals_or_y_pred, dtype=float)
    if len(y_t) == 0:
        return 0.0
    
    # If binary 0/1 inputs
    if np.all(np.isin(y_t, [0, 1])) and np.all(np.isin(y_p, [0, 1])):
        return float(np.mean(y_t == y_p))
    
    # Continuous signals and returns: positive matches positive, negative matches negative
    dir_t = np.where(y_t >= 0, 1, -1)
    dir_p = np.where(y_p >= 0, 1, -1)
    return float(np.mean(dir_t == dir_p))


def compute_precision_recall_f1(y_true: Union[List, np.ndarray],
                                y_pred: Union[List, np.ndarray]) -> Dict[str, float]:
    """Computes precision, recall, and F1 score with zero_division safety."""
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    if len(y_t) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    prec = float(precision_score(y_t, y_p, zero_division=0))
    rec = float(recall_score(y_t, y_p, zero_division=0))
    f1 = float(f1_score(y_t, y_p, zero_division=0))
    return {"precision": prec, "recall": rec, "f1": f1}


def compute_roc_auc(y_true: Union[List, np.ndarray],
                    y_prob: Union[List, np.ndarray]) -> float:
    """Computes ROC-AUC score, handling single-class edge cases."""
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_prob)
    if len(y_t) == 0 or len(np.unique(y_t)) < 2:
        return 0.5
    try:
        return float(roc_auc_score(y_t, y_p))
    except Exception:
        return 0.5


def compute_brier_score(y_true: Union[List, np.ndarray],
                        y_prob: Union[List, np.ndarray]) -> float:
    """Computes Brier Score (mean squared error between probabilities and binary labels)."""
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_prob, dtype=float)
    if len(y_t) == 0:
        return 0.0
    return float(np.mean((y_p - y_t) ** 2))


def compute_log_loss(y_true: Union[List, np.ndarray],
                     y_prob: Union[List, np.ndarray],
                     eps: float = 1e-15) -> float:
    """Computes binary cross-entropy / log loss with probability clipping."""
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.clip(np.asarray(y_prob, dtype=float), eps, 1 - eps)
    if len(y_t) == 0:
        return 0.0
    if len(np.unique(y_t)) < 2:
        # Direct formula for single class
        loss = -np.mean(y_t * np.log(y_p) + (1 - y_t) * np.log(1 - y_p))
        return float(loss)
    return float(log_loss(y_t, y_p))


def compute_ece(y_true: Union[List, np.ndarray],
                y_prob: Union[List, np.ndarray],
                n_bins: int = 10) -> float:
    """
    Computes Expected Calibration Error (ECE):
    Partitions predictions into M equal-width bins, calculates absolute difference
    between average confidence and empirical accuracy weighted by bin frequency.
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_prob, dtype=float)
    n = len(y_t)
    if n == 0:
        return 0.0

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        if i == n_bins - 1:
            in_bin = (y_p >= bin_lower) & (y_p <= bin_upper)
        else:
            in_bin = (y_p >= bin_lower) & (y_p < bin_upper)
            
        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_acc = np.mean(y_t[in_bin])
            bin_conf = np.mean(y_p[in_bin])
            ece += (bin_size / n) * np.abs(bin_acc - bin_conf)

    return float(ece)


def compute_spearman_rank_ic(signals: Union[List, np.ndarray],
                             forward_returns: Union[List, np.ndarray]) -> float:
    """Computes Spearman Rank Correlation (Information Coefficient) between signals and forward returns."""
    sig = np.asarray(signals, dtype=float)
    ret = np.asarray(forward_returns, dtype=float)
    if len(sig) < 3 or np.std(sig) == 0 or np.std(ret) == 0:
        return 0.0
    corr, _ = stats.spearmanr(sig, ret)
    if np.isnan(corr):
        return 0.0
    return float(corr)


def compute_ic_series_statistics(ic_series: Union[List, np.ndarray]) -> Dict[str, float]:
    """
    Computes summary statistics for a time-series of Information Coefficients (IC):
      - IC Mean
      - IC Standard Deviation
      - IC t-statistic
      - Information Ratio (IR = mean / std * sqrt(252))
    """
    ics = np.asarray(ic_series, dtype=float)
    ics = ics[~np.isnan(ics)]
    n = len(ics)
    if n < 2:
        mean_ic = float(ics[0]) if n == 1 else 0.0
        return {
            "ic_mean": mean_ic,
            "ic_std": 0.0,
            "ic_t_stat": 0.0,
            "information_ratio": 0.0,
            "observations": n
        }
    
    mean_ic = float(np.mean(ics))
    std_ic = float(np.std(ics, ddof=1))
    if std_ic == 0.0:
        t_stat = 0.0
        ir = 0.0
    else:
        t_stat = float(mean_ic / (std_ic / np.sqrt(n)))
        ir = float((mean_ic / std_ic) * np.sqrt(252))

    return {
        "ic_mean": mean_ic,
        "ic_std": std_ic,
        "ic_t_stat": t_stat,
        "information_ratio": ir,
        "observations": n
    }


# =====================================================================
# 2. FINANCIAL & PERFORMANCE METRICS
# =====================================================================

def compute_annualized_return(returns: Union[List, np.ndarray], periods_per_year: int = 252) -> float:
    """Computes compound annualized return from a sequence of periodic returns."""
    rets = np.asarray(returns, dtype=float)
    if len(rets) == 0:
        return 0.0
    cum_ret = np.prod(1.0 + rets)
    n_years = len(rets) / periods_per_year
    if n_years <= 0:
        return 0.0
    try:
        ann_ret = (cum_ret ** (1.0 / n_years)) - 1.0
        return float(ann_ret)
    except Exception:
        return float(np.mean(rets) * periods_per_year)


def compute_sharpe_ratio(returns: Union[List, np.ndarray],
                         risk_free_rate: float = 0.0,
                         periods_per_year: int = 252) -> float:
    """
    Computes annualized Sharpe Ratio.
    risk_free_rate is annualized (e.g. 0.05 for 5%).
    """
    rets = np.asarray(returns, dtype=float)
    if len(rets) < 2:
        return 0.0
    rf_per_period = risk_free_rate / periods_per_year
    excess_rets = rets - rf_per_period
    std = np.std(excess_rets, ddof=1)
    if std == 0.0:
        return 0.0
    sharpe = (np.mean(excess_rets) / std) * np.sqrt(periods_per_year)
    return float(sharpe)


def compute_sortino_ratio(returns: Union[List, np.ndarray],
                          risk_free_rate: float = 0.0,
                          periods_per_year: int = 252) -> float:
    """
    Computes annualized Sortino Ratio (downside deviation as penalty).
    """
    rets = np.asarray(returns, dtype=float)
    if len(rets) < 2:
        return 0.0
    rf_per_period = risk_free_rate / periods_per_year
    excess_rets = rets - rf_per_period
    downside = np.minimum(0.0, excess_rets)
    downside_std = np.sqrt(np.mean(downside ** 2))
    if downside_std == 0.0:
        return 0.0
    sortino = (np.mean(excess_rets) / downside_std) * np.sqrt(periods_per_year)
    return float(sortino)


def compute_max_drawdown(equity_or_returns: Union[List, np.ndarray], is_returns: bool = False) -> float:
    """
    Computes maximum peak-to-trough drawdown as a positive decimal (e.g., 0.15 = -15% drawdown).
    """
    arr = np.asarray(equity_or_returns, dtype=float)
    if len(arr) == 0:
        return 0.0
    if is_returns:
        cum_equity = np.cumprod(1.0 + arr)
    else:
        cum_equity = arr
    
    running_max = np.maximum.accumulate(cum_equity)
    drawdowns = (running_max - cum_equity) / np.maximum(running_max, 1e-9)
    return float(np.max(drawdowns))


def compute_calmar_ratio(annualized_return: float, max_drawdown: float) -> float:
    """Computes Calmar Ratio: annualized return / max drawdown."""
    if max_drawdown <= 0.0:
        return 0.0
    return float(annualized_return / max_drawdown)


def compute_profit_factor(trade_returns_or_pnls: Union[List, np.ndarray]) -> float:
    """
    Computes Profit Factor: Gross Profits / Gross Losses.
    Returns 99.0 if there are zero losses and positive gains.
    """
    pnl = np.asarray(trade_returns_or_pnls, dtype=float)
    if len(pnl) == 0:
        return 0.0
    gains = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    total_gains = np.sum(gains) if len(gains) > 0 else 0.0
    total_losses = np.abs(np.sum(losses)) if len(losses) > 0 else 0.0

    if total_losses == 0.0:
        return 99.0 if total_gains > 0.0 else 0.0
    return float(total_gains / total_losses)


def compute_win_loss_ratio(trade_returns_or_pnls: Union[List, np.ndarray]) -> Dict[str, float]:
    """
    Computes Win Rate and Win/Loss Ratio (Average Gain / Average Loss).
    """
    pnl = np.asarray(trade_returns_or_pnls, dtype=float)
    if len(pnl) == 0:
        return {"hit_rate": 0.0, "win_loss_ratio": 0.0, "avg_win": 0.0, "avg_loss": 0.0}
    
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    hit_rate = len(wins) / len(pnl)
    avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
    avg_loss = float(np.abs(np.mean(losses))) if len(losses) > 0 else 0.0
    wl_ratio = (avg_win / avg_loss) if avg_loss > 0 else (99.0 if avg_win > 0 else 0.0)

    return {
        "hit_rate": float(hit_rate),
        "win_loss_ratio": float(wl_ratio),
        "avg_win": avg_win,
        "avg_loss": avg_loss
    }


def compute_alpha(portfolio_returns: Union[List, np.ndarray],
                  benchmark_returns: Union[List, np.ndarray],
                  risk_free_rate: float = 0.0,
                  periods_per_year: int = 252) -> Dict[str, float]:
    """
    Computes Jensen's Alpha and Beta against a benchmark series.
    """
    rp = np.asarray(portfolio_returns, dtype=float)
    rb = np.asarray(benchmark_returns, dtype=float)
    min_len = min(len(rp), len(rb))
    if min_len < 3:
        return {"alpha": 0.0, "beta": 1.0, "r_squared": 0.0}
    
    rp = rp[:min_len]
    rb = rb[:min_len]
    
    rf_period = risk_free_rate / periods_per_year
    excess_p = rp - rf_period
    excess_b = rb - rf_period

    var_b = np.var(excess_b, ddof=1)
    if var_b == 0:
        return {"alpha": 0.0, "beta": 1.0, "r_squared": 0.0}
    
    cov = np.cov(excess_p, excess_b)[0, 1]
    beta = float(cov / var_b)
    
    # Annualized Alpha
    alpha_period = np.mean(excess_p) - beta * np.mean(excess_b)
    alpha_ann = float(alpha_period * periods_per_year)

    corr = np.corrcoef(excess_p, excess_b)[0, 1]
    r_squared = float(corr ** 2) if not np.isnan(corr) else 0.0

    return {
        "alpha": alpha_ann,
        "beta": beta,
        "r_squared": r_squared
    }


# =====================================================================
# 3. COMPREHENSIVE SUITE BUNDLE
# =====================================================================

def evaluate_predictions_full(y_true: Union[List, np.ndarray],
                             y_prob: Union[List, np.ndarray],
                             threshold: float = 0.50,
                             forward_returns: Optional[Union[List, np.ndarray]] = None) -> Dict[str, Any]:
    """
    Runs both statistical and financial evaluations on a set of model predictions.
    """
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_prob, dtype=float)
    preds = (y_p >= threshold).astype(int)

    clf_metrics = compute_precision_recall_f1(y_t, preds)
    acc = compute_accuracy(y_t, preds)
    roc = compute_roc_auc(y_t, y_p)
    brier = compute_brier_score(y_t, y_p)
    logl = compute_log_loss(y_t, y_p)
    ece = compute_ece(y_t, y_p)

    out = {
        "sample_size": len(y_t),
        "threshold": threshold,
        "accuracy": acc,
        "precision": clf_metrics["precision"],
        "recall": clf_metrics["recall"],
        "f1": clf_metrics["f1"],
        "roc_auc": roc,
        "brier_score": brier,
        "log_loss": logl,
        "ece": ece,
    }

    if forward_returns is not None and len(forward_returns) == len(y_t):
        rets = np.asarray(forward_returns, dtype=float)
        rank_ic = compute_spearman_rank_ic(y_p, rets)
        out["rank_ic"] = rank_ic
        
        # Strategy returns: long when predicted 1, flat when 0
        strat_rets = np.where(preds == 1, rets, 0.0)
        ann_ret = compute_annualized_return(strat_rets)
        sharpe = compute_sharpe_ratio(strat_rets)
        sortino = compute_sortino_ratio(strat_rets)
        mdd = compute_max_drawdown(strat_rets, is_returns=True)
        calmar = compute_calmar_ratio(ann_ret, mdd)
        pf = compute_profit_factor(strat_rets[strat_rets != 0])
        wl = compute_win_loss_ratio(strat_rets[strat_rets != 0])

        out.update({
            "annualized_return": ann_ret,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": mdd,
            "calmar_ratio": calmar,
            "profit_factor": pf,
            "hit_rate": wl["hit_rate"],
            "win_loss_ratio": wl["win_loss_ratio"]
        })

    return out
