"""
Centralized Risk Engine & Capital Protection Guard for Sprint 10.
Production Readiness, Risk Monitoring & Paper-Trading Reliability.

Enforces:
1. Configurable risk limits (single position, sector exposure, cash reserve, drawdown).
2. Pre-trade order validation independent of analyst and committee enthusiasm.
3. Persistent Global Trading Kill Switch with audit trail and state recovery.
4. Non-liquidation safety: blocks new trading activity safely without forced firesales.
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger("RiskGuard")

STATE_FILE = Path(__file__).resolve().parent.parent / "data" / "kill_switch_state.json"


@dataclass
class RiskLimits:
    """Configurable institutional risk parameters."""
    max_single_position_pct: float = 0.15     # Max 15% in any single ticker
    max_sector_exposure_pct: float = 0.40     # Max 40% in any economic sector
    min_cash_pct: float = 0.10                # Minimum 10% cash liquidity reserve
    max_portfolio_drawdown_pct: float = 0.20  # Max 20% peak-to-trough drawdown threshold
    max_daily_loss_pct: float = 0.05          # Max 5% single-day loss threshold
    paper_trading_only: bool = True           # Non-negotiable safety invariant for Sprint 10


@dataclass
class RiskValidationResult:
    """Structured result returned by pre-trade validation gate."""
    approved: bool
    reason: str
    violation_code: Optional[str] = None
    requested_dollars: float = 0.0
    allowed_dollars: float = 0.0
    adjusted_quantity: float = 0.0


class RiskGuard:
    """Centralized risk management engine with persistent kill switch."""

    def __init__(self, limits: Optional[RiskLimits] = None, state_file: Optional[Path] = None):
        self.limits = limits or RiskLimits()
        self.state_file = state_file or STATE_FILE
        self._ensure_state_file()

    def _ensure_state_file(self):
        """Ensures the kill switch state file exists with default inactive state."""
        if not self.state_file.exists():
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            default_state = {
                "active": False,
                "reason": "Default initialized",
                "activated_at": None,
                "activated_by": None,
                "history": []
            }
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(default_state, f, indent=2)

    def _read_state(self) -> Dict[str, Any]:
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading kill switch state: {e}. Defaulting to SAFE ACTIVE mode.")
            return {"active": True, "reason": f"State read error: {e}", "activated_at": None, "activated_by": "SYSTEM"}

    def _write_state(self, state: Dict[str, Any]):
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    def is_kill_switch_active(self) -> Tuple[bool, Dict[str, Any]]:
        """Returns whether the global kill switch is currently engaged and the active metadata."""
        state = self._read_state()
        return bool(state.get("active", False)), state

    def activate_kill_switch(self, reason: str, triggered_by: str = "SYSTEM") -> Dict[str, Any]:
        """
        Activates the global trading kill switch.
        Blocks all new order validation and execution immediately.
        """
        state = self._read_state()
        now_iso = datetime.now(timezone.utc).isoformat()
        event = {
            "action": "ACTIVATE",
            "reason": reason,
            "by": triggered_by,
            "timestamp": now_iso
        }
        state["active"] = True
        state["reason"] = reason
        state["activated_at"] = now_iso
        state["activated_by"] = triggered_by
        if "history" not in state:
            state["history"] = []
        state["history"].append(event)
        self._write_state(state)
        logger.warning(f"GLOBAL TRADING KILL SWITCH ACTIVATED by {triggered_by}. Reason: {reason}")
        return state

    def deactivate_kill_switch(self, reason: str, approved_by: str = "RISK_COMMITTEE") -> Dict[str, Any]:
        """
        Deactivates the global trading kill switch following verified risk approval.
        """
        state = self._read_state()
        now_iso = datetime.now(timezone.utc).isoformat()
        event = {
            "action": "DEACTIVATE",
            "reason": reason,
            "by": approved_by,
            "timestamp": now_iso
        }
        state["active"] = False
        state["reason"] = f"Deactivated: {reason}"
        state["activated_at"] = None
        state["activated_by"] = None
        if "history" not in state:
            state["history"] = []
        state["history"].append(event)
        self._write_state(state)
        logger.info(f"GLOBAL TRADING KILL SWITCH DEACTIVATED by {approved_by}. Reason: {reason}")
        return state

    def validate_order(
        self,
        ticker: str,
        side: str,
        quantity: float,
        price: float,
        portfolio_state: Dict[str, Any],
        sector: str = "Technology",
        is_stale: bool = False
    ) -> RiskValidationResult:
        """
        Pre-trade risk evaluation gate. Must pass before order generation.
        """
        side = side.upper().strip()
        ticker = ticker.upper().strip()

        # 1. Kill Switch Check
        kill_active, k_state = self.is_kill_switch_active()
        if kill_active:
            return RiskValidationResult(
                approved=False,
                reason=f"KILL_SWITCH_ACTIVE: {k_state.get('reason')}",
                violation_code="KILL_SWITCH_ENGAGED",
                requested_dollars=quantity * price
            )

        # 2. Stale Market Data Check
        if is_stale:
            return RiskValidationResult(
                approved=False,
                reason="STALE_DATA_REJECTION: Market price feed is stale or missing",
                violation_code="STALE_MARKET_DATA",
                requested_dollars=quantity * price
            )

        # 3. Parameter Sanity
        if quantity <= 0.0 or price <= 0.0:
            return RiskValidationResult(
                approved=False,
                reason=f"INVALID_PARAMETERS: quantity={quantity}, price={price}",
                violation_code="INVALID_QUANTITY_PRICE"
            )

        total_equity = float(portfolio_state.get("total_equity", 0.0))
        cash = float(portfolio_state.get("cash", 0.0))
        open_positions = portfolio_state.get("open_positions", {})
        peak_equity = float(portfolio_state.get("peak_equity", total_equity))

        if total_equity <= 0.0:
            return RiskValidationResult(
                approved=False,
                reason="ZERO_EQUITY_REJECTION: Portfolio total equity is non-positive",
                violation_code="INSOLVENT_PORTFOLIO"
            )

        # 4. Maximum Drawdown Check
        current_drawdown = (peak_equity - total_equity) / peak_equity if peak_equity > 0 else 0.0
        if current_drawdown >= self.limits.max_portfolio_drawdown_pct:
            return RiskValidationResult(
                approved=False,
                reason=f"MAX_DRAWDOWN_EXCEEDED: Current drawdown {current_drawdown:.1%} >= limit {self.limits.max_portfolio_drawdown_pct:.1%}",
                violation_code="DRAWDOWN_LIMIT_BREACH"
            )

        # For SELL orders, verify sufficient open position to close
        if side == "SELL":
            existing_shares = open_positions.get(ticker, {}).get("shares", 0.0)
            if quantity > existing_shares + 1e-5:
                return RiskValidationResult(
                    approved=False,
                    reason=f"SHORT_SALE_RESTRICTION: Attempted to sell {quantity:.2f} shares, but own {existing_shares:.2f}",
                    violation_code="UNCOVERED_SHORT_SALE"
                )
            return RiskValidationResult(
                approved=True,
                reason="SELL order risk approved",
                requested_dollars=quantity * price,
                allowed_dollars=quantity * price,
                adjusted_quantity=quantity
            )

        # BUY Order Evaluation:
        order_cost = quantity * price
        # Account for 10 bps friction buffer (0.10%)
        required_cash = order_cost * 1.0010

        # 5. Cash Liquidity Reserve Check
        min_cash_required = total_equity * self.limits.min_cash_pct
        available_cash_for_orders = max(0.0, cash - min_cash_required)

        if required_cash > available_cash_for_orders:
            # Check if partial allocation is possible above minimum threshold ($100 USD)
            if available_cash_for_orders < 100.0:
                return RiskValidationResult(
                    approved=False,
                    reason=f"CASH_RESERVE_VIOLATION: Required {required_cash:,.2f} exceeds available unreserved cash {available_cash_for_orders:,.2f} (Reserve: {min_cash_required:,.2f})",
                    violation_code="INSUFFICIENT_UNRESERVED_CASH",
                    requested_dollars=order_cost,
                    allowed_dollars=0.0
                )
            # Size down to available cash
            order_cost = available_cash_for_orders / 1.0010
            quantity = order_cost / price

        # 6. Single Position Concentration Limit Check
        existing_pos = open_positions.get(ticker, {})
        existing_val = existing_pos.get("shares", 0.0) * existing_pos.get("current_price", price)
        max_allowed_ticker_val = total_equity * self.limits.max_single_position_pct
        if (existing_val + order_cost) > max_allowed_ticker_val:
            allowed_additional = max_allowed_ticker_val - existing_val
            if allowed_additional < 100.0:
                return RiskValidationResult(
                    approved=False,
                    reason=f"MAX_POSITION_CONCENTRATION: Ticker {ticker} exposure ({existing_val + order_cost:,.2f}) exceeds {self.limits.max_single_position_pct:.0%} of equity ({max_allowed_ticker_val:,.2f})",
                    violation_code="SINGLE_POSITION_LIMIT_EXCEEDED",
                    requested_dollars=quantity * price,
                    allowed_dollars=0.0
                )
            order_cost = allowed_additional
            quantity = order_cost / price

        # 7. Sector Exposure Concentration Limit Check
        sector_val = 0.0
        for t_k, pos in open_positions.items():
            pos_sec = pos.get("sector", "Technology")
            if pos_sec == sector:
                sector_val += pos.get("shares", 0.0) * pos.get("current_price", price)

        max_allowed_sector_val = total_equity * self.limits.max_sector_exposure_pct
        if (sector_val + order_cost) > max_allowed_sector_val:
            allowed_additional_sector = max_allowed_sector_val - sector_val
            if allowed_additional_sector < 100.0:
                return RiskValidationResult(
                    approved=False,
                    reason=f"MAX_SECTOR_EXPOSURE: Sector '{sector}' exposure ({sector_val + order_cost:,.2f}) exceeds {self.limits.max_sector_exposure_pct:.0%} of equity ({max_allowed_sector_val:,.2f})",
                    violation_code="SECTOR_LIMIT_EXCEEDED",
                    requested_dollars=quantity * price,
                    allowed_dollars=0.0
                )
            order_cost = allowed_additional_sector
            quantity = order_cost / price

        return RiskValidationResult(
            approved=True,
            reason="Pre-trade risk limits satisfied",
            requested_dollars=quantity * price,
            allowed_dollars=order_cost,
            adjusted_quantity=quantity
        )
