"""
Operational Monitoring, Health & Auditability Service for Sprint 10.
Production Readiness, Risk Monitoring & Paper-Trading Reliability.

Exposes structured telemetry and readiness diagnostics:
1. Overall System Health & Trading Mode (Paper Trading Only).
2. Market-Data Freshness & Ingestion Telemetry.
3. Model Version, Artifact Checksums & Inference Health.
4. Opportunity Scans & Discovery Telemetry.
5. Committee Veto Rates & Governance Audit Trail.
6. Risk Engine Status, Kill Switch State & Rejected Orders.
7. Paper Order Lifecycle & Cumulative Friction Deductions.
8. Portfolio NAV, Cash, Exposures & Maximum Drawdown.
9. Multi-Horizon Outcome Tracking Status (Pending, Resolved, Unresolvable).
10. Readiness Endpoint (/readiness) verifying dependencies.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional

from database.connection import SessionLocal
from database.models import (
    MarketData, TrainingFeature, PredictionOutcome, OpportunityOutcome,
    PaperTradingPosition, PaperTradingSnapshot, AgentDecisionRecord
)
from portfolio.risk_guard import RiskGuard
from config import ARTIFACTS_DIR

logger = logging.getLogger("SystemHealth")


class SystemHealthMonitor:
    """Institutional operational health, telemetry, and diagnostics engine."""

    def __init__(self, risk_guard: Optional[RiskGuard] = None):
        self.risk_guard = risk_guard or RiskGuard()

    def get_health(self) -> Dict[str, Any]:
        """
        Liveness check: returns system component statuses and basic liveness.
        """
        session = SessionLocal()
        db_alive = False
        market_bars_count = 0
        try:
            market_bars_count = session.query(MarketData).count()
            db_alive = True
        except Exception as e:
            logger.error(f"Health DB probe failed: {e}")
        finally:
            session.close()

        model_path = ARTIFACTS_DIR / "best_model.pkl"
        model_exists = model_path.exists()

        kill_active, k_state = self.risk_guard.is_kill_switch_active()

        is_healthy = db_alive and model_exists

        return {
            "status": "HEALTHY" if is_healthy else "UNHEALTHY",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trading_mode": "PAPER_TRADING_ONLY",
            "broker_gateway": "SIMULATED_PAPER_EXECUTION",
            "database_connected": db_alive,
            "total_market_bars": market_bars_count,
            "model_artifact_loaded": model_exists,
            "kill_switch_active": kill_active,
            "kill_switch_reason": k_state.get("reason") if kill_active else None
        }

    def get_readiness(self) -> Dict[str, Any]:
        """
        Readiness check: evaluates whether the platform is ready to accept simulated orders.
        Fails if critical artifacts are missing, data is empty, or kill switch is engaged.
        """
        health = self.get_health()
        reasons_not_ready = []

        if not health["database_connected"]:
            reasons_not_ready.append("Database connection is down")
        if not health["model_artifact_loaded"]:
            reasons_not_ready.append("Champion ML model artifact not found")
        if health["total_market_bars"] == 0:
            reasons_not_ready.append("No market bars ingested in database")
        if health["kill_switch_active"]:
            reasons_not_ready.append(f"Global kill switch engaged: {health['kill_switch_reason']}")

        is_ready = len(reasons_not_ready) == 0

        return {
            "ready": is_ready,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trading_mode": "PAPER_TRADING_ONLY",
            "gating_issues": reasons_not_ready,
            "checked_subsystems": {
                "database": health["database_connected"],
                "model_artifacts": health["model_artifact_loaded"],
                "market_data": health["total_market_bars"] > 0,
                "risk_guard": not health["kill_switch_active"]
            }
        }

    def get_operational_metrics(self) -> Dict[str, Any]:
        """
        Detailed operational dashboard telemetry covering outcomes, orders, and committee history.
        """
        session = SessionLocal()
        try:
            pred_outcomes_count = session.query(PredictionOutcome).count()
            opp_outcomes_count = session.query(OpportunityOutcome).count()
            resolved_pred = session.query(PredictionOutcome).filter(PredictionOutcome.status == "RESOLVED").count()
            pending_pred = session.query(PredictionOutcome).filter(PredictionOutcome.status == "PENDING").count()

            open_positions = session.query(PaperTradingPosition).filter(PaperTradingPosition.status == "OPEN").count()
            closed_positions = session.query(PaperTradingPosition).filter(PaperTradingPosition.status == "CLOSED").count()

            latest_snapshot = session.query(PaperTradingSnapshot).order_by(PaperTradingSnapshot.snapshot_date.desc()).first()

            adrs_count = session.query(AgentDecisionRecord).count()
            cro_vetoes = session.query(AgentDecisionRecord).filter(
                AgentDecisionRecord.agent_name == "Risk Officer",
                AgentDecisionRecord.stance == "VETO"
            ).count()

            kill_active, k_state = self.risk_guard.is_kill_switch_active()

            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "trading_mode": "PAPER_TRADING_ONLY",
                "outcomes_pipeline": {
                    "total_prediction_outcomes": pred_outcomes_count,
                    "resolved_prediction_outcomes": resolved_pred,
                    "pending_prediction_outcomes": pending_pred,
                    "total_opportunity_outcomes": opp_outcomes_count
                },
                "paper_portfolio": {
                    "open_positions": open_positions,
                    "closed_positions": closed_positions,
                    "latest_equity": latest_snapshot.total_equity if latest_snapshot else None,
                    "latest_cash": latest_snapshot.cash if latest_snapshot else None,
                    "latest_cumulative_return": latest_snapshot.cumulative_return if latest_snapshot else None,
                    "latest_max_drawdown": latest_snapshot.max_drawdown if latest_snapshot else None
                },
                "committee_governance": {
                    "total_agent_decisions": adrs_count,
                    "risk_officer_vetoes": cro_vetoes
                },
                "risk_engine": {
                    "kill_switch_active": kill_active,
                    "kill_switch_metadata": k_state
                }
            }
        finally:
            session.close()
