"""
Broker-Neutral Paper Execution Gateway for Sprint 10.
Production Readiness, Risk Monitoring & Paper-Trading Reliability.

Features:
1. Standardized institutional interface:
   - submit_order()
   - get_order()
   - cancel_order()
   - get_positions()
   - get_account_state()
2. Strict Order Lifecycle State Machine:
   CREATED -> VALIDATED -> SUBMITTED -> FILLED / REJECTED / CANCELLED
3. Idempotency Key deduplication (prevents duplicate orders after retries/restarts).
4. Pre-trade Risk Enforcement via RiskGuard.
5. Atomic Cash & Holdings ledger execution with single-deduction friction.
6. USD Base Currency Accounting for US Equities.
"""

import uuid
import logging
import threading
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timezone
from enum import Enum
from typing import Dict, List, Any, Optional

from portfolio.risk_guard import RiskGuard, RiskValidationResult
from portfolio.ledger import PortfolioLedger, PositionRecord
from portfolio.transaction_costs import TransactionCostModel
from execution.order_store import OrderStore

logger = logging.getLogger("PaperGateway")


class OrderStatus(str, Enum):
    CREATED = "CREATED"
    VALIDATED = "VALIDATED"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


@dataclass
class OrderRequest:
    """Input payload to request order execution."""
    ticker: str
    side: str  # "BUY" or "SELL"
    quantity: float
    order_type: str = "MARKET"
    limit_price: Optional[float] = None
    strategy_id: str = "STRAT_ML_COMMITTEE_V1"
    trade_date: Optional[date] = None
    sector: str = "Technology"
    idempotency_key: Optional[str] = None
    trace_id: Optional[str] = None


@dataclass
class Order:
    """Full life-cycle institutional order record."""
    order_id: str
    idempotency_key: str
    ticker: str
    side: str
    quantity: float
    order_type: str
    status: OrderStatus
    created_at: str
    submitted_at: Optional[str] = None
    filled_at: Optional[str] = None
    reference_price: float = 0.0
    fill_price: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0   # informational: dollar slippage already embedded in fill_price
    spread: float = 0.0
    total_fees: float = 0.0  # cash fees charged to the ledger = commission + spread
    rejection_reason: Optional[str] = None
    strategy_id: str = "STRAT_ML_COMMITTEE_V1"
    trace_id: Optional[str] = None
    state_history: List[Dict[str, Any]] = field(default_factory=list)
    trade_date: Optional[str] = None
    sector: str = "Technology"


def _order_to_dict(order: Order) -> Dict[str, Any]:
    d = asdict(order)
    d["status"] = order.status.value
    return d


def _order_from_dict(d: Dict[str, Any]) -> Order:
    d = dict(d)
    d["status"] = OrderStatus(d["status"])
    return Order(**d)


class PaperExecutionGateway:
    """
    Broker-neutral simulated execution gateway enforcing strict order states and risk checks.

    Cost accounting (each cost applied exactly once):
      - slippage: embedded in the fill price (BUY fills above, SELL below the reference
        price by `cost_model.slippage_rate`); NOT charged again as a cash fee.
      - commission and spread: computed on fill-price notional and charged as cash fees.
    """

    # Allowed state machine transitions
    VALID_TRANSITIONS = {
        OrderStatus.CREATED: {OrderStatus.VALIDATED, OrderStatus.REJECTED},
        OrderStatus.VALIDATED: {OrderStatus.SUBMITTED, OrderStatus.REJECTED},
        OrderStatus.SUBMITTED: {OrderStatus.FILLED, OrderStatus.REJECTED, OrderStatus.CANCELLED},
        OrderStatus.FILLED: set(),
        OrderStatus.REJECTED: set(),
        OrderStatus.CANCELLED: set(),
    }

    def __init__(
        self,
        ledger: Optional[PortfolioLedger] = None,
        risk_guard: Optional[RiskGuard] = None,
        cost_model: Optional[TransactionCostModel] = None,
        store: Optional[OrderStore] = None
    ):
        self.ledger = ledger or PortfolioLedger()
        self.risk_guard = risk_guard or RiskGuard()
        self.cost_model = cost_model or TransactionCostModel()
        self.store = store
        self._lock = threading.RLock()
        self.orders: Dict[str, Order] = {}
        self.orders_by_idempotency: Dict[str, str] = {}  # idempotency_key -> order_id
        if self.store is not None:
            self._restore_from_store()

    def _persist(self, order: Order) -> None:
        if self.store is not None:
            self.store.update(_order_to_dict(order))

    def _restore_from_store(self) -> None:
        """
        Restart recovery: reload all orders and replay FILLED ones into a fresh ledger
        so cash/positions match what was already executed (no duplicate fills).
        """
        fresh_ledger = not self.ledger.transaction_history
        for d in self.store.all_orders():
            order = _order_from_dict(d)
            self.orders[order.order_id] = order
            self.orders_by_idempotency[order.idempotency_key] = order.order_id
            if order.status == OrderStatus.FILLED and fresh_ledger:
                costs = {"commission": order.commission, "slippage": 0.0, "spread": order.spread}
                trade_date = date.fromisoformat(order.trade_date) if order.trade_date else date.today()
                if order.side == "BUY":
                    self.ledger.record_buy(order.ticker, order.quantity, order.fill_price,
                                           trade_date, costs, sector=order.sector, trace_id=order.trace_id)
                else:
                    self.ledger.record_sell(order.ticker, order.fill_price, trade_date, costs,
                                            trace_id=order.trace_id, shares=order.quantity)

    def _transition_status(self, order: Order, new_status: OrderStatus, reason: Optional[str] = None):
        """Validates and applies an order state transition."""
        allowed = self.VALID_TRANSITIONS.get(order.status, set())
        if new_status not in allowed:
            raise ValueError(
                f"Invalid order lifecycle transition: {order.status.value} -> {new_status.value} for order {order.order_id}"
            )
        order.status = new_status
        now_str = datetime.now(timezone.utc).isoformat()
        order.state_history.append({
            "status": new_status.value,
            "timestamp": now_str,
            "reason": reason
        })
        if new_status == OrderStatus.SUBMITTED:
            order.submitted_at = now_str
        elif new_status == OrderStatus.FILLED:
            order.filled_at = now_str
        elif new_status == OrderStatus.REJECTED:
            order.rejection_reason = reason
        self._persist(order)

    def submit_order(
        self,
        request: OrderRequest,
        market_price: float,
        is_stale: bool = False
    ) -> Order:
        """
        Executes the complete order submission pipeline (serialized by a lock):
        1. Idempotency deduplication check (memory, then durable store).
        2. Lifecycle creation (CREATED).
        3. Pre-trade risk validation gate (VALIDATED or REJECTED).
        4. Submission to paper matching engine (SUBMITTED).
        5. Execution fill against reference price + friction (FILLED).
        6. Atomic portfolio ledger update.
        """
        with self._lock:
            return self._submit_order_locked(request, market_price, is_stale)

    def _submit_order_locked(self, request: OrderRequest, market_price: float, is_stale: bool) -> Order:
        now_str = datetime.now(timezone.utc).isoformat()
        ticker = request.ticker.upper().strip()
        side = request.side.upper().strip()
        trade_date = request.trade_date or date.today()

        # 1. Idempotency Deduplication Check
        idempotency_key = request.idempotency_key or f"{trade_date}_{ticker}_{side}_{request.quantity}_{request.strategy_id}"
        if idempotency_key in self.orders_by_idempotency:
            existing_id = self.orders_by_idempotency[idempotency_key]
            logger.info(f"Duplicate order request with idempotency key '{idempotency_key}'. Returning existing order {existing_id}.")
            return self.orders[existing_id]
        if self.store is not None:
            persisted = self.store.get_by_key(idempotency_key)
            if persisted is not None:
                order = _order_from_dict(persisted)
                self.orders[order.order_id] = order
                self.orders_by_idempotency[idempotency_key] = order.order_id
                return order

        order_id = str(uuid.uuid4())
        trace_id = request.trace_id or str(uuid.uuid4())

        # 2. State: CREATED
        order = Order(
            order_id=order_id,
            idempotency_key=idempotency_key,
            ticker=ticker,
            side=side,
            quantity=request.quantity,
            order_type=request.order_type,
            status=OrderStatus.CREATED,
            created_at=now_str,
            reference_price=market_price,
            strategy_id=request.strategy_id,
            trace_id=trace_id,
            trade_date=trade_date.isoformat(),
            sector=request.sector
        )
        order.state_history.append({"status": OrderStatus.CREATED.value, "timestamp": now_str})
        if self.store is not None and not self.store.insert_if_new(_order_to_dict(order)):
            # Another process won the race on this key: return its order, never execute twice.
            persisted = self.store.get_by_key(idempotency_key)
            existing = _order_from_dict(persisted)
            self.orders[existing.order_id] = existing
            self.orders_by_idempotency[idempotency_key] = existing.order_id
            return existing
        self.orders[order_id] = order
        self.orders_by_idempotency[idempotency_key] = order_id

        # 3. Pre-Trade Risk Gate
        portfolio_state = {
            "total_equity": self.ledger.get_total_equity(),
            "cash": self.ledger.cash,
            "peak_equity": self.ledger.peak_equity,
            "day_start_equity": self.ledger.equity_at_start_of(trade_date),
            "open_positions": {
                t: {"shares": p.shares, "current_price": p.current_price, "sector": p.sector}
                for t, p in self.ledger.open_positions.items()
            }
        }

        risk_res: RiskValidationResult = self.risk_guard.validate_order(
            ticker=ticker,
            side=side,
            quantity=request.quantity,
            price=market_price,
            portfolio_state=portfolio_state,
            sector=request.sector,
            is_stale=is_stale
        )

        if not risk_res.approved:
            self._transition_status(order, OrderStatus.REJECTED, reason=risk_res.reason)
            logger.warning(f"Order {order_id} rejected by RiskGuard: {risk_res.reason}")
            return order

        # State: VALIDATED
        self._transition_status(order, OrderStatus.VALIDATED)
        if risk_res.adjusted_quantity > 0 and risk_res.adjusted_quantity != order.quantity:
            order.quantity = risk_res.adjusted_quantity

        # 4. State: SUBMITTED
        self._transition_status(order, OrderStatus.SUBMITTED)

        # 5. Simulated fill. Slippage is embedded in the fill price (applied once);
        #    commission and spread are cash fees on the fill notional.
        slip_rate = self.cost_model.slippage_rate
        fill_price = market_price * (1.0 + slip_rate) if side == "BUY" else market_price * (1.0 - slip_rate)
        gross_value = order.quantity * fill_price
        costs = self.cost_model.compute_trade_cost(gross_value, side=side)

        order.fill_price = fill_price
        order.commission = costs["commission"]
        order.spread = costs["spread"]
        order.slippage = order.quantity * abs(fill_price - market_price)
        order.total_fees = costs["commission"] + costs["spread"]
        ledger_costs = {"commission": costs["commission"], "slippage": 0.0, "spread": costs["spread"]}

        # 6. Atomic Portfolio Ledger Update (ledger validates before mutating)
        try:
            if side == "BUY":
                self.ledger.record_buy(
                    ticker=ticker,
                    shares=order.quantity,
                    fill_price=fill_price,
                    trade_date=trade_date,
                    costs=ledger_costs,
                    sector=request.sector,
                    trace_id=trace_id
                )
            else:
                self.ledger.record_sell(
                    ticker=ticker,
                    fill_price=fill_price,
                    trade_date=trade_date,
                    costs=ledger_costs,
                    trace_id=trace_id,
                    shares=order.quantity
                )

            # State: FILLED
            self._transition_status(order, OrderStatus.FILLED)
            logger.info(
                f"Order {order_id} FILLED: {side} {order.quantity:.2f} {ticker} @ {fill_price:.2f} USD "
                f"(Fees: {order.total_fees:.2f} USD, Trace: {trace_id})"
            )
        except Exception as e:
            # Ledger validates before mutating, so no balance was changed.
            logger.error(f"Execution error during ledger recording for order {order_id}: {e}")
            self._transition_status(order, OrderStatus.REJECTED, reason=f"Ledger error: {e}")

        return order


    def get_order(self, order_id: str) -> Optional[Order]:
        """Fetches order record by ID."""
        return self.orders.get(order_id)

    def cancel_order(self, order_id: str, reason: str = "Client requested cancellation") -> bool:
        """Cancels a pending SUBMITTED order (if not already FILLED)."""
        order = self.orders.get(order_id)
        if not order:
            return False
        if order.status == OrderStatus.SUBMITTED:
            self._transition_status(order, OrderStatus.CANCELLED, reason=reason)
            return True
        return False

    def get_positions(self) -> Dict[str, Dict[str, Any]]:
        """Returns current open positions from the ledger."""
        return {
            ticker: {
                "ticker": pos.ticker,
                "shares": pos.shares,
                "entry_price": pos.entry_price,
                "current_price": pos.current_price,
                "market_value": pos.market_value,
                "unrealized_pnl": pos.unrealized_pnl,
                "sector": pos.sector,
                "entry_date": pos.entry_date.isoformat() if hasattr(pos.entry_date, "isoformat") else str(pos.entry_date)
            }
            for ticker, pos in self.ledger.open_positions.items()
        }

    def get_account_state(self) -> Dict[str, Any]:
        """Returns the current account state with exact USD equity and cash balances."""
        return {
            "currency": self.ledger.currency,
            "total_equity": self.ledger.get_total_equity(),
            "cash": self.ledger.cash,
            "positions_value": self.ledger.get_positions_value(),
            "open_positions_count": len(self.ledger.open_positions),
            "closed_trades_count": len(self.ledger.closed_positions),
            "peak_equity": self.ledger.peak_equity,
            "kill_switch_active": self.risk_guard.is_kill_switch_active()[0]
        }
