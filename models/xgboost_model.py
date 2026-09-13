import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from config import (
    ARTIFACTS_DIR,
    TRAIN_TEST_SPLIT_RATIO,
    XGB_N_ESTIMATORS,
    XGB_LEARNING_RATE,
    XGB_MAX_DEPTH,
)
from features.builder import FEATURE_COLUMNS, TARGET_COLUMN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("XGBoostModel")

MODEL_PATH = ARTIFACTS_DIR / "xgboost_v1.json"
META_PATH = ARTIFACTS_DIR / "xgboost_v1_meta.joblib"


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    params: Dict[str, Any] = None,
) -> Tuple[xgb.XGBClassifier, np.ndarray, np.ndarray]:
    """
    Trains an XGBoost classifier with custom or default parameters.
    """
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = (neg_count / (pos_count + 1e-5)) if pos_count > 0 else 1.0

    default_params = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": scale_pos_weight,
        "eval_metric": "logloss",
        "random_state": 42,
    }
    if params:
        default_params.update(params)

    model = xgb.XGBClassifier(**default_params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False,
    )

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return model, y_pred, y_prob


def train_xgboost_pipeline(
    df: pd.DataFrame,
    features: list = None,
    target: str = TARGET_COLUMN,
    split_ratio: float = TRAIN_TEST_SPLIT_RATIO,
) -> Tuple[xgb.XGBClassifier, Dict[str, Any], pd.DataFrame]:
    """
    Trains an XGBoost classifier with strict temporal split to prevent lookahead bias.
    Evaluates on out-of-sample forward data.
    """
    if features is None:
        features = FEATURE_COLUMNS

    df_clean = df.sort_values(by=["date"]).reset_index(drop=True)

    X = df_clean[features]
    y = df_clean[target]

    split_idx = int(len(df_clean) * split_ratio)
    if split_idx <= 0 or split_idx >= len(df_clean):
        raise ValueError("Insufficient data points for temporal train/test split.")

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    test_df = df_clean.iloc[split_idx:].copy()

    logger.info(
        f"Temporal Split: Train={len(X_train)} samples ({df_clean['date'].iloc[0]} to {df_clean['date'].iloc[split_idx-1]}), "
        f"Test={len(X_test)} samples ({df_clean['date'].iloc[split_idx]} to {df_clean['date'].iloc[-1]})"
    )

    # Class balance weighting
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = (neg_count / (pos_count + 1e-5)) if pos_count > 0 else 1.0

    model = xgb.XGBClassifier(
        n_estimators=XGB_N_ESTIMATORS,
        learning_rate=XGB_LEARNING_RATE,
        max_depth=XGB_MAX_DEPTH,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False,
    )

    # Predictions & Probabilities
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    test_df["prediction"] = y_pred
    test_df["confidence"] = y_prob

    # Performance Metrics
    metrics = evaluate_model(y_test, y_pred, y_prob)

    # Save Artifacts
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_PATH))
    joblib.dump({"features": features, "metrics": metrics}, META_PATH)
    logger.info(f"Model saved to {MODEL_PATH}")

    return model, metrics, test_df


def evaluate_model(y_true, y_pred, y_prob) -> Dict[str, Any]:
    """Computes standard ML classification metrics and trading win rate."""
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = 0.5

    cm = confusion_matrix(y_true, y_pred).tolist()

    # Buy Signal Win Rate = Precision of BUY class (class 1)
    win_rate = prec

    metrics = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(auc, 4),
        "win_rate": round(win_rate, 4),
        "confusion_matrix": cm,
    }

    logger.info(
        f"Evaluation Results -> Acc: {metrics['accuracy']:.2%}, "
        f"Win Rate (Prec): {metrics['win_rate']:.2%}, "
        f"F1: {metrics['f1']:.4f}, AUC: {metrics['roc_auc']:.4f}"
    )
    return metrics


def load_xgboost_model() -> Tuple[xgb.XGBClassifier, Dict[str, Any]]:
    """Loads saved XGBoost model and metadata."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))
    meta = joblib.load(META_PATH) if META_PATH.exists() else {}
    return model, meta
