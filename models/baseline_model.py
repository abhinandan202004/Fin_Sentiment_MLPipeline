import logging
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("BaselineModel")

DEFAULT_BASELINE_FEATURES = [
    "avg_sentiment",
    "total_article_count",
    "rsi",
    "macd",
    "ema20",
    "volume_ratio",
]


def train_logistic_regression_baseline(
    df: pd.DataFrame,
    features: List[str] = None,
    target_col: str = "target",
    train_ratio: float = 0.80,
) -> Tuple[LogisticRegression, Dict[str, Any], pd.DataFrame]:
    """
    Trains a Logistic Regression benchmark using a strict temporal split (no shuffle).
    Evaluates out-of-sample directional accuracy and classification performance.
    """
    if features is None:
        features = DEFAULT_BASELINE_FEATURES

    # Ensure temporal ordering
    df_sorted = df.sort_values(by="date").reset_index(drop=True)

    # Drop any rows with NaN in features or target
    clean_df = df_sorted.dropna(subset=features + [target_col]).copy()

    split_idx = int(len(clean_df) * train_ratio)
    if split_idx <= 0 or split_idx >= len(clean_df):
        raise ValueError("Insufficient data points for temporal split.")

    train_data = clean_df.iloc[:split_idx]
    test_data = clean_df.iloc[split_idx:].copy()

    X_train_raw = train_data[features]
    y_train = train_data[target_col].astype(int)

    X_test_raw = test_data[features]
    y_test = test_data[target_col].astype(int)

    logger.info(
        f"Temporal Split: Train={len(train_data)} ({train_data['date'].iloc[0]} to {train_data['date'].iloc[-1]}), "
        f"Test={len(test_data)} ({test_data['date'].iloc[0]} to {test_data['date'].iloc[-1]})"
    )

    # Feature Standardization (fit on train only to prevent data leakage)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # Train Logistic Regression (clean default l2 regularization)
    clf = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    # Out-of-sample predictions
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    test_data["pred_baseline"] = y_pred
    test_data["prob_baseline"] = y_prob

    # Metrics
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    try:
        auc = float(roc_auc_score(y_test, y_prob))
    except Exception:
        auc = 0.5

    cm = confusion_matrix(y_test, y_pred).tolist()
    clf_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    # Primary KPI: Directional Accuracy
    directional_accuracy = acc

    # Feature Coefficients
    coef_dict = {feat: round(float(coef), 4) for feat, coef in zip(features, clf.coef_[0])}

    metrics = {
        "accuracy": round(acc, 4),
        "directional_accuracy": round(directional_accuracy, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(auc, 4),
        "confusion_matrix": cm,
        "feature_coefficients": coef_dict,
        "classification_report": clf_report,
    }

    print("\n" + "-" * 60)
    print(" LOGISTIC REGRESSION BASELINE EVALUATION (Out-of-Sample)")
    print("-" * 60)
    print(f"  Directional Accuracy : {metrics['directional_accuracy']:.2%}  (Target: >52%)")
    print(f"  Precision (BUY)      : {metrics['precision']:.2%}")
    print(f"  Recall (BUY)         : {metrics['recall']:.2%}")
    print(f"  F1 Score             : {metrics['f1']:.4f}")
    print(f"  ROC-AUC              : {metrics['roc_auc']:.4f}")
    print(f"  Confusion Matrix     : {cm}")
    print("\n  Learned Feature Weights (Standardized Odds Ratio Drivers):")
    for feat, coef in coef_dict.items():
        print(f"    - {feat:<20}: {coef:+.4f}")
    print("-" * 60 + "\n")

    return clf, metrics, test_data
