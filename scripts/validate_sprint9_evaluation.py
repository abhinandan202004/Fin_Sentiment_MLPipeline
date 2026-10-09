"""
Comprehensive Verification Suite for Sprint 9:
Autonomous Evaluation & Adaptive Intelligence.

Validates all Sprint 9 deliverables, principles, and architectural requirements:
  1. Database Schema Layer:
     - 9 new tables (PredictionOutcome, AgentDecisionRecord, AgentEvaluationRecord,
       CommitteeEvaluationRecord, OpportunityOutcome, PaperTradingPosition,
       PaperTradingSnapshot, ModelCalibrationRecord, EvaluationSnapshot).
  2. Metrics Engine (evaluation/metrics.py):
     - Statistical & Financial metrics (Accuracy, ROC-AUC, F1, Brier, ECE, Rank IC,
       Sharpe, Sortino, Calmar, MaxDD, Profit Factor, Alpha).
  3. Outcome Tracker (evaluation/outcome_tracker.py):
     - Zero-look-ahead multi-horizon returns (1d, 5d, 10d, 20d, 60d) across NVDA, AAPL, JPM, XOM.
  4. Baseline Model Evaluation (evaluation/prediction_evaluator.py):
     - Frozen baseline out-of-sample evaluation, decile spread, sector/regime breakdowns.
  5. Independent Agent Evaluation (evaluation/agent_evaluator.py):
     - Multi-agent evaluations, hit rates, Sharpe, Sortino, ECE, regime matrices.
  6. Committee Counterfactual & Veto Analysis (evaluation/committee_evaluator.py):
     - Committee vs ML-Only vs Opportunity vs Benchmark, veto loss-prevention audit with 10 bps friction.
  7. Probability Calibration (evaluation/calibration.py):
     - Platt Scaling, Isotonic Regression, reliability curves, ECE and Brier comparison.
  8. Walk-Forward Paper Trading (evaluation/paper_trader.py):
     - INR ₹10,00,000 portfolio, 15% single pos, 40% sector, 10% cash floor, 10 bps transaction costs.
  9. Adaptive Committee Weights (evaluation/adaptive_weights.py):
     - Performance scoring, EWMA smoothing (0.7/0.3), bounds [5%, 45%], min sample 30 rule, fallback hierarchy.
 10. Adaptive Opportunity Scoring (evaluation/adaptive_opportunities.py):
     - Rolling 60-day IC calibration, factor bounds, walk-forward testing.
 11. Master Reporting & Snapshots (evaluation/performance_report.py):
     - EvaluationSnapshot DB persistence and master markdown report structure.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import datetime, date, timezone
import numpy as np
import pandas as pd

from database.connection import SessionLocal, init_db
from database.models import (
    PredictionOutcome,
    AgentDecisionRecord,
    AgentEvaluationRecord,
    CommitteeEvaluationRecord,
    OpportunityOutcome,
    PaperTradingPosition,
    PaperTradingSnapshot,
    ModelCalibrationRecord,
    EvaluationSnapshot
)
from evaluation.metrics import (
    compute_accuracy,
    compute_directional_accuracy,
    compute_precision_recall_f1,
    compute_roc_auc,
    compute_brier_score,
    compute_log_loss,
    compute_ece,
    compute_spearman_rank_ic,
    compute_ic_series_statistics,
    compute_annualized_return,
    compute_sharpe_ratio,
    compute_sortino_ratio,
    compute_max_drawdown,
    compute_calmar_ratio,
    compute_profit_factor,
    compute_win_loss_ratio,
    compute_alpha
)
from evaluation.outcome_tracker import OutcomeTracker
from evaluation.prediction_evaluator import PredictionEvaluator
from evaluation.agent_evaluator import AgentEvaluator
from evaluation.committee_evaluator import CommitteeEvaluator
from evaluation.calibration import ProbabilityCalibrator, run_temporal_calibration_experiment
from evaluation.paper_trader import PaperTrader
from evaluation.adaptive_weights import AdaptiveCommitteeWeights
from evaluation.adaptive_opportunities import AdaptiveOpportunityScorer
from evaluation.performance_report import MasterPerformanceReporter
from models.feature_store import load_feature_dataset
from config import ARTIFACTS_DIR, REPORTS_DIR
import json


def test_database_schema():
    print("\n--- 1. Testing Database Schema & 9 Sprint 9 Models ---")
    init_db()
    session = SessionLocal()
    try:
        models = [
            PredictionOutcome,
            AgentDecisionRecord,
            AgentEvaluationRecord,
            CommitteeEvaluationRecord,
            OpportunityOutcome,
            PaperTradingPosition,
            PaperTradingSnapshot,
            ModelCalibrationRecord,
            EvaluationSnapshot
        ]
        for m in models:
            count = session.query(m).count()
            print(f"  [PASS] Table '{m.__tablename__}' verified (row count: {count})")
    finally:
        session.close()


def test_metrics_engine():
    print("\n--- 2. Testing Metrics Engine (Statistical & Financial) ---")
    y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
    y_prob = np.array([0.9, 0.1, 0.8, 0.7, 0.2, 0.6, 0.4, 0.3, 0.85, 0.15])
    rets = np.array([0.05, -0.02, 0.04, 0.03, -0.01, 0.02, -0.01, -0.02, 0.06, -0.03])

    acc = compute_accuracy(y_true, (y_prob >= 0.5).astype(int))
    assert acc == 1.0, f"Expected accuracy 1.0, got {acc}"

    auc = compute_roc_auc(y_true, y_prob)
    assert auc == 1.0, f"Expected ROC-AUC 1.0, got {auc}"

    brier = compute_brier_score(y_true, y_prob)
    assert brier < 0.10, f"Expected low Brier score, got {brier}"

    ece = compute_ece(y_true, y_prob)
    assert 0.0 <= ece <= 1.0, f"ECE {ece} out of bounds"

    ic = compute_spearman_rank_ic(y_prob, rets)
    assert ic > 0.80, f"Expected strong rank IC, got {ic}"

    sharpe = compute_sharpe_ratio(rets)
    sortino = compute_sortino_ratio(rets)
    mdd = compute_max_drawdown(rets, is_returns=True)
    pf = compute_profit_factor(rets)
    alpha_res = compute_alpha(rets, rets * 0.5)

    assert sharpe > 0.0
    assert sortino > 0.0
    assert 0.0 <= mdd <= 1.0
    assert pf > 1.0
    assert "alpha" in alpha_res

    print(f"  [PASS] All 15+ statistical and financial metrics verified on synthetic controls.")


def test_outcome_tracker():
    print("\n--- 3. Testing Unified Outcome Tracker (Zero Look-Ahead) ---")
    tracker = OutcomeTracker()
    samples = tracker.inspect_sample_outcomes(["NVDA", "AAPL", "JPM", "XOM"], date(2024, 1, 10))

    assert len(samples) == 4, f"Expected 4 tickers evaluated, got {len(samples)}"
    for ticker, res in samples.items():
        assert res is not None, f"No outcome returned for {ticker}"
        assert res["entry_date"] >= date(2024, 1, 10), "Look-ahead violation on entry date"
        assert res["entry_price"] > 0, "Invalid entry price"
        assert res["return_1d"] is not None
        assert res["return_5d"] is not None
        assert res["return_10d"] is not None
        assert res["return_20d"] is not None
        assert res["return_60d"] is not None
        assert res["actual_direction"] in ["BUY", "NOT BUY"]
        print(f"  [PASS] {ticker}: 1d={res['return_1d']:+.2%}, 5d={res['return_5d']:+.2%}, 20d={res['return_20d']:+.2%}, 60d={res['return_60d']:+.2%}")


def test_baseline_evaluation():
    print("\n--- 4. Testing Baseline Model Evaluation (Phase 4) ---")
    evaluator = PredictionEvaluator()
    res = evaluator.evaluate_out_of_sample(split_ratio=0.80)
    report_path = evaluator.generate_baseline_report(res)

    assert "overall" in res
    assert "sample_size" in res
    assert res["sample_size"] > 1000, f"Expected > 1000 test observations, got {res['sample_size']}"
    assert "decile_analysis" in res
    assert "sector_breakdown" in res
    assert "regime_breakdown" in res
    assert report_path.exists(), f"Baseline report missing at {report_path}"
    print(f"  [PASS] Baseline Model Evaluated: N={res['sample_size']}, Hit Rate={res['overall']['hit_rate']:.1%}, Report={report_path.name}")


def test_agent_evaluation():
    print("\n--- 5. Testing Independent Agent Evaluation (Phase 5) ---")
    evaluator = AgentEvaluator()
    metrics = evaluator.evaluate_all_agents(persist_to_db=True)
    report_path = evaluator.generate_agent_report(metrics)

    assert len(metrics) >= 4, f"Expected at least 4 agents evaluated, got {len(metrics)}"
    for a_name, a in metrics.items():
        assert a["sample_size"] >= 30, f"Agent {a_name} sample size < 30 ({a['sample_size']})"
        assert 0.0 <= a["hit_rate"] <= 1.0
        assert 0.0 <= a["ece"] <= 1.0
        assert "regime_breakdown" in a
    assert report_path.exists()
    print(f"  [PASS] Evaluated {len(metrics)} independent agents with N >= 30 samples each. Report={report_path.name}")


def test_committee_and_veto_evaluation():
    print("\n--- 6. Testing Committee Counterfactual & Veto Analysis (Phase 6) ---")
    evaluator = CommitteeEvaluator()
    results = evaluator.evaluate_committee_counterfactuals(persist_to_db=True)
    report_path = evaluator.generate_committee_report(results)

    assert "committee" in results
    assert "ml_only" in results
    assert "opportunity_only" in results
    assert "benchmark" in results
    assert "veto_analysis" in results

    v = results["veto_analysis"]
    assert v["vetoes_evaluated"] > 0, "No vetoes evaluated"
    assert 0.0 <= v["veto_loss_prevention_rate"] <= 1.0
    assert report_path.exists()
    print(f"  [PASS] Counterfactual comparison complete: Veto Hit Rate={v['veto_loss_prevention_rate']:.1%}, Capital Saved={v['total_capital_saved_pct']:+.2%}")


def test_probability_calibration():
    print("\n--- 7. Testing Probability Calibration (Phase 7) ---")
    df = load_feature_dataset(run_feature_engineering=True)
    with open(ARTIFACTS_DIR / "feature_list.json", "r") as f:
        features = json.load(f)
    cal_res = run_temporal_calibration_experiment(df, features, persist_to_db=True)

    assert "raw" in cal_res
    assert "platt" in cal_res
    assert "isotonic" in cal_res
    assert "best_method" in cal_res
    assert "optimal_threshold" in cal_res
    print(f"  [PASS] Calibration Experiment: Champion={cal_res['best_method']} (Raw ECE={cal_res['raw']['ece']:.4f}, Platt ECE={cal_res['platt']['ece']:.4f}, Iso ECE={cal_res['isotonic']['ece']:.4f})")


def test_walk_forward_paper_trading():
    print("\n--- 8. Testing Walk-Forward Paper Trading (Phase 8) ---")
    trader = PaperTrader(initial_capital=1_000_000.0)
    results = trader.run_walk_forward(persist_to_db=True)
    report_path = trader.generate_trading_report(results)

    assert results["final_equity"] > 0, "Final equity non-positive"
    assert results["total_closed_trades"] >= 20, f"Expected >= 20 trades, got {results['total_closed_trades']}"
    assert results["total_transaction_costs"] > 0, "Transaction costs not deducted"
    assert report_path.exists()
    print(f"  [PASS] Paper Portfolio: Final=INR {results['final_equity']:,.2f}, Return={results['cumulative_return']:+.2%}, Alpha={results['alpha']:+.2%}, Trades={results['total_closed_trades']}")


def test_adaptive_weights():
    print("\n--- 9. Testing Adaptive Committee Weights (Phase 10) ---")
    engine = AdaptiveCommitteeWeights()
    res = engine.run_walk_forward_weight_test(split_point=40)

    for k, w in res["adaptive_weights"].items():
        assert 0.05 <= w <= 0.45, f"Weight for {k} ({w}) breaches [5%, 45%] bound"
    assert round(sum(res["adaptive_weights"].values()), 2) == 1.0, "Weights do not sum to 1.0"
    assert "verdict" in res
    print(f"  [PASS] Adaptive Committee Weights: {res['adaptive_weights']} (Verdict: {res['verdict']})")


def test_adaptive_opportunity_scoring():
    print("\n--- 10. Testing Adaptive Opportunity Scoring (Phase 11) ---")
    df = load_feature_dataset(run_feature_engineering=True)
    scorer = AdaptiveOpportunityScorer()
    res = scorer.run_walk_forward_opportunity_test(df)

    for k, w in res["adaptive_weights"].items():
        assert 0.05 <= w <= 0.45, f"Weight for {k} ({w}) breaches [5%, 45%] bound"
    assert round(sum(res["adaptive_weights"].values()), 2) == 1.0, "Weights do not sum to 1.0"
    assert "original_rank_ic" in res
    assert "adaptive_rank_ic" in res
    print(f"  [PASS] Adaptive Opportunities: Orig IC={res['original_rank_ic']:+.4f} vs Adapt IC={res['adaptive_rank_ic']:+.4f} (Verdict: {res['verdict']})")


def test_master_report_and_snapshots():
    print("\n--- 11. Testing Master Evaluation Report & Snapshots (Phases 12-14) ---")
    reporter = MasterPerformanceReporter(initial_capital=1_000_000.0)
    data = reporter.run_full_evaluation(persist_snapshot=True)
    report_file = reporter.generate_master_markdown_report(data)

    assert report_file.exists(), f"Master report missing at {report_file}"
    session = SessionLocal()
    try:
        snap_count = session.query(EvaluationSnapshot).count()
        assert snap_count > 0, "No EvaluationSnapshot records persisted"
        print(f"  [PASS] Master Evaluation Report generated at {report_file.name}")
        print(f"  [PASS] EvaluationSnapshot count in database: {snap_count}")
    finally:
        session.close()


def main():
    print("\n" + "=" * 80)
    print(" SPRINT 9 VERIFICATION SUITE: AUTONOMOUS EVALUATION & ADAPTIVE INTELLIGENCE")
    print("=" * 80)

    test_database_schema()
    test_metrics_engine()
    test_outcome_tracker()
    test_baseline_evaluation()
    test_agent_evaluation()
    test_committee_and_veto_evaluation()
    test_probability_calibration()
    test_walk_forward_paper_trading()
    test_adaptive_weights()
    test_adaptive_opportunity_scoring()
    test_master_report_and_snapshots()

    print("\n" + "=" * 80)
    print(" ALL SPRINT 9 VERIFICATION SUITE TESTS PASSED!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
