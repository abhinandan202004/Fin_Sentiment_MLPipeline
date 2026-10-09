"""
CLI Script for Phase 7: Probability Calibration.
Sprint 9: Autonomous Evaluation & Adaptive Intelligence.

Executes temporal probability calibration (Train -> Calibrate -> Evaluate)
and optimizes out-of-sample ECE & Brier score.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from config import ARTIFACTS_DIR
from models.feature_store import load_feature_dataset
from evaluation.calibration import run_temporal_calibration_experiment

def main():
    print("\n=======================================================")
    print(" SPRINT 9: TEMPORAL PROBABILITY CALIBRATION")
    print("=======================================================\n")

    with open(ARTIFACTS_DIR / "feature_list.json", "r") as f:
        features = json.load(f)

    df = load_feature_dataset(run_feature_engineering=True)
    res = run_temporal_calibration_experiment(df, features, persist_to_db=True)

    print(f"Evaluation Window:       {res['eval_window']}")
    print(f"Sample Size:             {res['sample_size']} observations")
    print("-" * 55)
    print(f"Raw Probabilities:       ECE = {res['raw']['ece']:.4f} | Brier = {res['raw']['brier']:.4f}")
    print(f"Platt Scaling:           ECE = {res['platt']['ece']:.4f} | Brier = {res['platt']['brier']:.4f} (Thresh: {res['platt']['optimal_threshold']:.3f})")
    print(f"Isotonic Regression:     ECE = {res['isotonic']['ece']:.4f} | Brier = {res['isotonic']['brier']:.4f} (Thresh: {res['isotonic']['optimal_threshold']:.3f})")
    print("-" * 55)
    print(f"Champion Calibration:    {res['best_method']}")
    print(f"ECE Reduction:           {res['ece_reduction_pct']:+.1%}")
    print(f"Optimal Threshold:       {res['optimal_threshold']:.3f}")
    print("\n=======================================================\n")

if __name__ == "__main__":
    main()
