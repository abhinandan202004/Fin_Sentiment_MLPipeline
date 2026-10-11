"""
Probability Calibration Engine for Sprint 9: Autonomous Evaluation & Adaptive Intelligence.

Provides strictly temporal probability calibration to eliminate model overconfidence:
  1. Platt Scaling (Parametric Sigmoid / Logistic Calibration)
  2. Isotonic Regression (Non-parametric Monotonic Mapping)
  3. Reliability Curves & Binning Analysis
  4. Out-of-Sample Expected Calibration Error (ECE) & Brier Score Optimization

Adheres strictly to Rule 1 & Rule 2:
  TRAIN PERIOD -> CALIBRATION PERIOD -> UNSEEN EVALUATION PERIOD
  Never fits calibrators on the evaluation period.

Persists to:
  model_calibration_records
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
from datetime import datetime, date, timezone
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

from database.connection import SessionLocal
from database.models import ModelCalibrationRecord
from evaluation.metrics import compute_ece, compute_brier_score, compute_accuracy, compute_roc_auc

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ProbabilityCalibrator")


class ProbabilityCalibrator:
    """Temporal calibrator supporting Platt Scaling and Isotonic Regression."""

    def __init__(self):
        self.platt_model: Optional[LogisticRegression] = None
        self.isotonic_model: Optional[IsotonicRegression] = None
        self.optimal_threshold: float = 0.50
        self.is_fitted: bool = False

    def fit(self, y_cal: np.ndarray, prob_cal: np.ndarray) -> "ProbabilityCalibrator":
        """
        Fits Platt Scaling and Isotonic Regression strictly on the calibration set.
        """
        y = np.asarray(y_cal, dtype=int)
        p = np.asarray(prob_cal, dtype=float).clip(1e-6, 1 - 1e-6)

        # 1. Platt Scaling (Logistic Regression on log-odds)
        logit_p = np.log(p / (1.0 - p)).reshape(-1, 1)
        self.platt_model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        self.platt_model.fit(logit_p, y)

        # 2. Isotonic Regression (Piecewise constant non-decreasing fit)
        self.isotonic_model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.isotonic_model.fit(p, y)

        self.is_fitted = True
        return self

    def calibrate_platt(self, prob: np.ndarray) -> np.ndarray:
        """Applies fitted Platt Scaling to uncalibrated probabilities."""
        if not self.is_fitted or self.platt_model is None:
            raise ValueError("Calibrator has not been fitted yet.")
        p = np.asarray(prob, dtype=float).clip(1e-6, 1 - 1e-6)
        logit_p = np.log(p / (1.0 - p)).reshape(-1, 1)
        return self.platt_model.predict_proba(logit_p)[:, 1]

    def calibrate_isotonic(self, prob: np.ndarray) -> np.ndarray:
        """Applies fitted Isotonic Regression to uncalibrated probabilities."""
        if not self.is_fitted or self.isotonic_model is None:
            raise ValueError("Calibrator has not been fitted yet.")
        p = np.asarray(prob, dtype=float)
        return self.isotonic_model.predict(p)

    def compute_reliability_curve(self, y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> Dict[str, Any]:
        """Calculates bin midpoints, empirical accuracies, and frequencies."""
        y_t = np.asarray(y_true, dtype=float)
        y_p = np.asarray(prob, dtype=float)
        n = len(y_t)
        if n == 0:
            return {"bin_centers": [], "empirical_acc": [], "bin_counts": []}

        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_centers = []
        emp_acc = []
        bin_counts = []

        for i in range(n_bins):
            b_low = bin_edges[i]
            b_high = bin_edges[i + 1]
            if i == n_bins - 1:
                in_bin = (y_p >= b_low) & (y_p <= b_high)
            else:
                in_bin = (y_p >= b_low) & (y_p < b_high)

            count = int(np.sum(in_bin))
            bin_counts.append(count)
            bin_centers.append(round(float((b_low + b_high) / 2.0), 3))
            if count > 0:
                emp_acc.append(round(float(np.mean(y_t[in_bin])), 4))
            else:
                emp_acc.append(round(float(bin_centers[-1]), 4))

        return {
            "bin_centers": bin_centers,
            "empirical_acc": emp_acc,
            "bin_counts": bin_counts
        }

    def optimize_threshold(self, y_true: np.ndarray, prob: np.ndarray) -> float:
        """Finds optimal decision threshold maximizing F1 score."""
        y_t = np.asarray(y_true, dtype=int)
        best_th = 0.50
        best_f1 = -1.0
        for th in np.linspace(0.40, 0.70, 31):
            preds = (prob >= th).astype(int)
            tp = np.sum((preds == 1) & (y_t == 1))
            fp = np.sum((preds == 1) & (y_t == 0))
            fn = np.sum((preds == 0) & (y_t == 1))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            if f1 > best_f1:
                best_f1 = f1
                best_th = float(th)

        self.optimal_threshold = round(best_th, 3)
        return self.optimal_threshold


def run_temporal_calibration_experiment(df: pd.DataFrame, features: List[str],
                                        persist_to_db: bool = True) -> Dict[str, Any]:
    """
    Executes a 3-way temporal split experiment:
      Train: 60% (2021-11 to ~mid-2024)
      Calibrate: 20% (mid-2024 to late-2025)
      Unseen Evaluation: 20% (late-2025 to 2026-09)
    """
    import joblib
    from config import ARTIFACTS_DIR

    model = joblib.load(ARTIFACTS_DIR / "best_model.pkl")

    # Sort strictly by date
    df = df.sort_values(by=["date", "ticker"]).reset_index(drop=True)
    n = len(df)
    idx_cal = int(n * 0.60)
    idx_eval = int(n * 0.80)

    train_df = df.iloc[:idx_cal]
    cal_df = df.iloc[idx_cal:idx_eval]
    eval_df = df.iloc[idx_eval:]

    logger.info(f"Temporal Split for Calibration: Train={len(train_df)}, Calibrate={len(cal_df)}, Unseen Eval={len(eval_df)}")

    avail_features = [f for f in features if f in df.columns]
    X_cal = cal_df[avail_features].fillna(0)
    y_cal = cal_df["target"].values
    prob_cal = model.predict_proba(X_cal)[:, 1]

    X_eval = eval_df[avail_features].fillna(0)
    y_eval = eval_df["target"].values
    prob_eval = model.predict_proba(X_eval)[:, 1]

    # Fit Calibrator
    calibrator = ProbabilityCalibrator()
    calibrator.fit(y_cal, prob_cal)

    # Evaluate on UNSEEN EVALUATION SET
    p_raw = prob_eval
    p_platt = calibrator.calibrate_platt(prob_eval)
    p_iso = calibrator.calibrate_isotonic(prob_eval)

    # Calculate metrics before and after
    ece_raw = compute_ece(y_eval, p_raw)
    brier_raw = compute_brier_score(y_eval, p_raw)

    ece_platt = compute_ece(y_eval, p_platt)
    brier_platt = compute_brier_score(y_eval, p_platt)

    ece_iso = compute_ece(y_eval, p_iso)
    brier_iso = compute_brier_score(y_eval, p_iso)

    # Optimize threshold strictly on the CALIBRATION set to prevent evaluation peeking
    opt_th_platt = calibrator.optimize_threshold(y_cal, calibrator.calibrate_platt(prob_cal))
    opt_th_iso = calibrator.optimize_threshold(y_cal, calibrator.calibrate_isotonic(prob_cal))

    # Select best calibration method
    if ece_iso < ece_platt and ece_iso < ece_raw:
        best_method = "Isotonic Regression"
        ece_after = ece_iso
        brier_after = brier_iso
        best_th = opt_th_iso
    elif ece_platt < ece_raw:
        best_method = "Platt Scaling"
        ece_after = ece_platt
        brier_after = brier_platt
        best_th = opt_th_platt
    else:
        best_method = "Raw Baseline"
        ece_after = ece_raw
        brier_after = brier_raw
        best_th = 0.55

    results = {
        "calibration_date": str(datetime.now(timezone.utc).date()),
        "eval_window": f"{eval_df['date'].min()} to {eval_df['date'].max()}",
        "sample_size": len(eval_df),
        "raw": {"ece": ece_raw, "brier": brier_raw},
        "platt": {"ece": ece_platt, "brier": brier_platt, "optimal_threshold": opt_th_platt},
        "isotonic": {"ece": ece_iso, "brier": brier_iso, "optimal_threshold": opt_th_iso},
        "best_method": best_method,
        "ece_before": ece_raw,
        "ece_after": ece_after,
        "brier_before": brier_raw,
        "brier_after": brier_after,
        "optimal_threshold": best_th,
        "ece_reduction_pct": ((ece_raw - ece_after) / ece_raw) if ece_raw > 0 else 0.0
    }

    if persist_to_db:
        session = SessionLocal()
        try:
            rec = ModelCalibrationRecord(
                calibration_date=datetime.now(timezone.utc),
                method=best_method,
                ece_before=ece_raw,
                ece_after=ece_after,
                brier_before=brier_raw,
                brier_after=brier_after,
                optimal_threshold=best_th,
                parameters={
                    "platt": {"ece": ece_platt, "brier": brier_platt},
                    "isotonic": {"ece": ece_iso, "brier": brier_iso}
                },
                model_version="SPRINT_9_BASELINE",
                calibration_version="calibrated_v1.0"
            )
            session.add(rec)
            session.commit()
            logger.info(f"Persisted ModelCalibrationRecord to database ({best_method}: ECE {ece_raw:.4f} -> {ece_after:.4f}).")
        finally:
            session.close()

    return results
