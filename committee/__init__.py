"""
Investment Committee Governance & Debate Package for Fin_Sentiment_MLPipeline.

Exposes:
- DebateEngine: Master multi-agent debate orchestrator
- DecisionScorer: Weighted scoring and calibrated confidence
- GovernanceEngine: Compliance rules and risk veto triggers
- CommitteeMemory: Audit history and agent accuracy tracking
"""

from committee.governance_rules import GovernanceEngine, RISK_VETO_RULES
from committee.decision_score import DecisionScorer
from committee.committee_memory import CommitteeMemory

__all__ = [
    "GovernanceEngine",
    "RISK_VETO_RULES",
    "DecisionScorer",
    "CommitteeMemory",
]
