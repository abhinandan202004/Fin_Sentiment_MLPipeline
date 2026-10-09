"""
CLI Script for Phase 4: Baseline Model Evaluation.
Sprint 9: Autonomous Evaluation & Adaptive Intelligence.

Evaluates the existing ML model without modifications and outputs
reports/evaluation_reports/sprint9_baseline.md
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from evaluation.prediction_evaluator import PredictionEvaluator

def main():
    print("\n=======================================================")
    print(" SPRINT 9: BASELINE MODEL EVALUATION (FROZEN SPRINT 8)")
    print("=======================================================\n")

    evaluator = PredictionEvaluator()
    results = evaluator.evaluate_out_of_sample(split_ratio=0.80)
    report_file = evaluator.generate_baseline_report(results)

    ov = results["overall"]
    ic = results["ic_statistics"]

    print(f"Sample Size:         {results['sample_size']} observations")
    print(f"Test Window:         {results['test_window']}")
    print(f"Accuracy:            {ov['accuracy']:.1%}")
    print(f"ROC-AUC:             {ov['roc_auc']:.4f}")
    print(f"F1 Score:            {ov['f1']:.4f}")
    print(f"Brier Score:         {ov['brier_score']:.4f}")
    print(f"ECE:                 {ov['ece']:.4f}")
    print(f"Spearman Rank IC:    {ic['ic_mean']:+.4f} (t-stat: {ic['ic_t_stat']:+.2f})")
    print(f"Hit Rate (5d):       {ov.get('hit_rate', 0.0):.1%}")
    print(f"Sharpe Ratio:        {ov.get('sharpe_ratio', 0.0):.2f}")
    print(f"Sortino Ratio:       {ov.get('sortino_ratio', 0.0):.2f}")
    print(f"Max Drawdown:        {ov.get('max_drawdown', 0.0):.2%}")
    print(f"Profit Factor:       {ov.get('profit_factor', 0.0):.2f}")
    print(f"Report Generated:    {report_file}")
    print("\n=======================================================\n")

if __name__ == "__main__":
    main()
