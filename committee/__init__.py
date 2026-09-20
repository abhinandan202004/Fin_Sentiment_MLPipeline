"""
Investment Committee Governance & Debate Package for Fin_Sentiment_MLPipeline.

Exposes:
- DebateEngine: Master multi-agent debate orchestrator
- DecisionScorer: Weighted scoring and calibrated confidence
- GovernanceEngine: Compliance rules and risk veto triggers
- CommitteeMemory: Audit history and agent accuracy tracking

Sprint 7 — Learning & Attribution:
- OutcomeTracker: Realized return resolution via yfinance
- AgentAttribution: Per-agent accuracy, Brier score, calibration
- EvidenceAttribution: Per-signal-stream reliability
- RegimeAnalysis: Performance stratified by market regime
- AdaptiveVoting: EWMA-learned agent weights
- CounterfactualAnalysis: Committee vs simpler strategies
"""

from committee.governance_rules import GovernanceEngine, RISK_VETO_RULES
from committee.decision_score import DecisionScorer
from committee.committee_memory import CommitteeMemory
from committee.outcome_tracker import OutcomeTracker
from committee.agent_attribution import AgentAttribution
from committee.evidence_attribution import EvidenceAttribution
from committee.regime_analysis import RegimeAnalysis
from committee.adaptive_voting import AdaptiveVoting
from committee.counterfactual import CounterfactualAnalysis

__all__ = [
    "GovernanceEngine",
    "RISK_VETO_RULES",
    "DecisionScorer",
    "CommitteeMemory",
    "OutcomeTracker",
    "AgentAttribution",
    "EvidenceAttribution",
    "RegimeAnalysis",
    "AdaptiveVoting",
    "CounterfactualAnalysis",
]

