import logging
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("LightGBMModel")


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    params: Optional[Dict[str, Any]] = None,
) -> Tuple[LGBMClassifier, np.ndarray, np.ndarray]:
    """
    Trains a LightGBM gradient boosted decision tree classifier.
    """
    default_params = {
        "n_estimators": 200,
        "max_depth": 5,
        "num_leaves": 31,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "class_weight": "balanced",
        "random_state": 42,
        "verbose": -1,
        "n_jobs": -1,
    }
    if params:
        default_params.update(params)

    model = LGBMClassifier(**default_params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return model, y_pred, y_prob
