import logging
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("RandomForestModel")


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    params: Optional[Dict[str, Any]] = None,
) -> Tuple[RandomForestClassifier, np.ndarray, np.ndarray]:
    """
    Trains a Random Forest classifier with balanced class weighting.
    """
    default_params = {
        "n_estimators": 200,
        "max_depth": 6,
        "min_samples_split": 10,
        "min_samples_leaf": 5,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
    }
    if params:
        default_params.update(params)

    model = RandomForestClassifier(**default_params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return model, y_pred, y_prob
