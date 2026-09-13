import logging
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SHAPExplainer")


def get_tree_explainer(model: xgb.XGBClassifier) -> shap.TreeExplainer:
    """Initializes a SHAP TreeExplainer for the XGBoost model."""
    return shap.TreeExplainer(model)


def compute_shap_values(
    model: xgb.XGBClassifier,
    X: pd.DataFrame,
) -> np.ndarray:
    """Computes full matrix of SHAP values for the given feature dataframe."""
    explainer = get_tree_explainer(model)
    shap_vals = explainer.shap_values(X)
    return shap_vals


def get_global_feature_importance(
    model: xgb.XGBClassifier,
    X: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """
    Computes global feature importance based on mean absolute SHAP values.
    """
    shap_vals = compute_shap_values(model, X)
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    features = list(X.columns)

    importance = [
        {"feature": feat, "mean_abs_shap": round(float(val), 5)}
        for feat, val in zip(features, mean_abs_shap)
    ]
    importance.sort(key=lambda x: x["mean_abs_shap"], reverse=True)
    return importance


def explain_instance(
    model: xgb.XGBClassifier,
    feature_row: pd.Series,
    feature_names: List[str] = None,
) -> Dict[str, Any]:
    """
    Explains an individual prediction instance, categorizing top positive and negative drivers.
    """
    if feature_names is None:
        feature_names = list(feature_row.index)

    row_df = pd.DataFrame([feature_row[feature_names]])
    explainer = get_tree_explainer(model)
    shap_vals = explainer.shap_values(row_df)[0]
    base_val = float(explainer.expected_value) if hasattr(explainer, "expected_value") else 0.0

    drivers = []
    for feat, val, s_val in zip(feature_names, row_df.iloc[0], shap_vals):
        drivers.append({
            "feature": feat,
            "value": round(float(val), 4) if isinstance(val, (int, float, np.number)) else str(val),
            "shap_impact": round(float(s_val), 4),
        })

    positive_drivers = sorted(
        [d for d in drivers if d["shap_impact"] > 0],
        key=lambda x: x["shap_impact"],
        reverse=True,
    )
    negative_drivers = sorted(
        [d for d in drivers if d["shap_impact"] < 0],
        key=lambda x: x["shap_impact"],
    )

    return {
        "base_value": round(base_val, 4),
        "total_shap_sum": round(float(np.sum(shap_vals)), 4),
        "positive_drivers": positive_drivers,
        "negative_drivers": negative_drivers,
    }
