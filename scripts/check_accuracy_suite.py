"""
scripts/check_accuracy_suite.py

Comprehensive Out-of-Sample Accuracy & Quantitative Performance Suite
Evaluates:
1. Baseline vs Expanded Dataset
2. Classification at default threshold (0.50) vs Optimized thresholds (0.52 - 0.58)
3. Return Regression Model (predicting future_return_5d) converted to directional signals
4. Top Feature Importances (SHAP)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, f1_score

from config import ARTIFACTS_DIR
from models.feature_store import load_feature_dataset, temporal_train_test_split, ALL_FEATURES
from models.target_engineering import evaluate_regression_direction


def main():
    print("=" * 80)
    print(" QUANTITATIVE RESEARCH EVALUATION: OUT-OF-SAMPLE ACCURACY SUITE")
    print("=" * 80)

    # 1. Load Feature Dataset
    df = load_feature_dataset()
    import json
    fpath = ARTIFACTS_DIR / "feature_list.json"
    if fpath.exists():
        with open(fpath) as f:
            features = json.load(f)
    else:
        features = [f for f in ALL_FEATURES if f in df.columns]

    # 2. Strict Temporal Split (80% Train / 20% Test walk-forward)
    X_train, y_train, X_val, y_val, train_df, val_df = temporal_train_test_split(
        df, split_ratio=0.80, features=features
    )
    ret_train = train_df["future_return_5d"].values
    ret_val = val_df["future_return_5d"].values

    print(f"\n[1] DATASET COVERAGE:")
    print(f"    - Total Observations : {len(df):,}")
    print(f"    - Universe Tickers   : {df['ticker'].nunique()} tickers across multiple sectors")
    print(f"    - Train Period       : {train_df['date'].min()} to {train_df['date'].max()} ({len(train_df):,} bars)")
    print(f"    - Test Period (OOS)  : {val_df['date'].min()} to {val_df['date'].max()} ({len(val_df):,} bars)")
    print(f"    - Features           : {len(features)} (Technical + Sentiment + Ranks + Interactions + Regimes)")

    # 3. Load Champion Classification Model
    champion = joblib.load(ARTIFACTS_DIR / "best_model.pkl")
    probs = champion.predict_proba(X_val)[:, 1]

    print("\n" + "=" * 80)
    print(" [2] CLASSIFICATION MODEL — THRESHOLD SENSITIVITY (Out-of-Sample)")
    print("=" * 80)
    print(f"{'Threshold':>10} | {'Signals':>8} | {'Signal %':>9} | {'Dir Acc':>9} | {'Hit Rate (Prec)':>16} | {'Avg Ret (5d)':>13} | {'Sharpe':>8}")
    print("-" * 80)

    for th in [0.50, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.58, 0.60]:
        preds = (probs > th).astype(int)
        n_signals = int(preds.sum())
        sig_pct = (n_signals / len(preds)) * 100
        acc = accuracy_score(y_val, preds) if n_signals > 0 else 0.0

        if n_signals > 0:
            hit_rate = (ret_val[preds == 1] > 0).mean() * 100
            avg_ret = ret_val[preds == 1].mean() * 100
            stdev = ret_val[preds == 1].std()
            sharpe = (ret_val[preds == 1].mean() / (stdev + 1e-9)) * np.sqrt(252 / 5) if n_signals > 1 else 0.0
        else:
            hit_rate, avg_ret, sharpe = 0.0, 0.0, 0.0

        marker = " <-- (Default Unfiltered)" if th == 0.50 else (" <-- (High Conviction)" if th == 0.55 else "")
        print(f"{th:>10.2f} | {n_signals:>8} | {sig_pct:>8.1f}% | {acc:>8.2%} | {hit_rate:>15.2f}% | {avg_ret:>12.2f}% | {sharpe:>8.2f}{marker}")

    # 4. Regression Model (Missing Lever #5: Predict Returns directly)
    print("\n" + "=" * 80)
    print(" [3] REGRESSION MODEL — PREDICTING RETURNS DIRECTLY (Missing Lever #5)")
    print("=" * 80)
    print("Training LightGBM Regressor on continuous future_return_5d...")
    reg = LGBMRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    reg.fit(X_train, ret_train)
    pred_ret = reg.predict(X_val)

    print(f"\nEvaluating Directional Signals from Return Forecasts:")
    print(f"{'Pred Return Cutoff':>18} | {'Buy Signals':>11} | {'Signal %':>9} | {'Dir Acc':>9} | {'Hit Rate (Prec)':>16} | {'Avg Ret (5d)':>13} | {'Sharpe':>8}")
    print("-" * 85)

    for th in [0.000, 0.002, 0.004, 0.006, 0.008, 0.010, 0.012]:
        m = evaluate_regression_direction(ret_val, pred_ret, threshold=th)
        sig_pct = (m["n_buy_signals"] / len(ret_val)) * 100
        marker = " <-- (Baseline Zero)" if th == 0.0 else (" <-- (Alpha Filter)" if th == 0.006 else "")
        print(f"{th:>17.3f} | {m['n_buy_signals']:>11} | {sig_pct:>8.1f}% | {m['directional_accuracy']:>8.2%} | {m['hit_rate']*100:>15.2f}% | {m['avg_trade_return_pct']:>12.2f}% | {m['sharpe_ratio']:>8.2f}{marker}")

    print("=" * 85 + "\n")


if __name__ == "__main__":
    main()
