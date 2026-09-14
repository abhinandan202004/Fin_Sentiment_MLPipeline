"""
Investment Committee Agents for Fin_Sentiment_MLPipeline.

Exposes:
- EvidenceProsecutor: Adversarial data auditing & quality checks
- BullAnalyst: Upside opportunities, catalysts, and BUY defense
- BearAnalyst: Downside vulnerability, skepticism, and technical exhaustion
- RiskOfficer: Portfolio risk, sector caps, and institutional veto rules
- PortfolioManager: Position sizing, Half-Kelly optimization, and constraint enforcement
- ChiefInvestmentOfficer: Adjudication, conflict resolution, and final investment decisions
"""

from agents.committee.evidence_prosecutor import EvidenceProsecutor
from agents.committee.bull_analyst import BullAnalyst
from agents.committee.bear_analyst import BearAnalyst
from agents.committee.risk_officer import RiskOfficer
from agents.committee.portfolio_manager import PortfolioManager
from agents.committee.cio_agent import ChiefInvestmentOfficer

__all__ = [
    "EvidenceProsecutor",
    "BullAnalyst",
    "BearAnalyst",
    "RiskOfficer",
    "PortfolioManager",
    "ChiefInvestmentOfficer",
]
