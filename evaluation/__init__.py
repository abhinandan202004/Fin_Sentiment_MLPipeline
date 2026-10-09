"""
Evaluation & Adaptive Intelligence Engine.
Sprint 9: Autonomous Evaluation & Adaptive Intelligence.
"""

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
    compute_alpha,
    evaluate_predictions_full,
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

__all__ = [
    "OutcomeTracker",
    "PredictionEvaluator",
    "AgentEvaluator",
    "CommitteeEvaluator",
    "ProbabilityCalibrator",
    "run_temporal_calibration_experiment",
    "PaperTrader",
    "AdaptiveCommitteeWeights",
    "AdaptiveOpportunityScorer",
    "MasterPerformanceReporter",
    "evaluate_predictions_full",
]
