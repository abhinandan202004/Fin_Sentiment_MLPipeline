# Sprint 10 — Central Risk Engine & Capital Protection Report

**Audit Date:** 2026-10-10  
**Scope:** Centralized Risk Guard, Configurable Guardrails, and Global Trading Kill Switch  

---

## 1. Executive Summary

In Sprint 10 Phase 3, centralized risk management was decoupled from the analyst and committee layer and consolidated into [`portfolio/risk_guard.py`](file:///c:/Users/ABHI%20N%20P/OneDrive/Desktop/Fin_Sentiment_MLPipeline/portfolio/risk_guard.py).

The risk engine operates as a mandatory pre-trade validation gate: every proposed order must pass risk inspection before reaching the execution gateway. Analyst excitement or high committee conviction cannot override these guardrails.

---

## 2. Configurable Institutional Risk Limits

| Risk Control | Default Limit | Purpose & Enforcement Method | Failure Action |
| :--- | :---: | :--- | :--- |
| **Max Single Position** | `15%` | Prevents idiosyncratic single-stock concentration. | Sized down to 15% headroom, or rejected if headroom < $100. |
| **Max Sector Exposure** | `40%` | Prevents correlated sector clustering (e.g. Technology over-concentration). | Sized down to 40% headroom, or rejected if headroom < $100. |
| **Minimum Cash Reserve**| `10%` | Preserves liquidity buffer for operational buffer and margin safety. | Order rejected with `INSUFFICIENT_UNRESERVED_CASH`. |
| **Max Portfolio Drawdown**| `20%` | Peak-to-trough equity protection limit. | All new BUY orders blocked with `DRAWDOWN_LIMIT_BREACH`. |
| **Max Daily Loss** | `5%` | Single-session volatility protection limit. | New BUY orders blocked with `DAILY_LOSS_LIMIT_BREACH`. |
| **Stale Data Block** | Active | Protects against execution during market data feed outages. | Order rejected with `STALE_MARKET_DATA`. |
| **Paper-Trading Mode** | Enforced | Non-negotiable invariant: prohibits real-money broker routing. | Live routing blocked by design. |

---

## 3. Persistent Global Trading Kill Switch

### Architecture & Persistence:
- **Persistent State:** Managed in [`data/kill_switch_state.json`](file:///c:/Users/ABHI%20N%20P/OneDrive/Desktop/Fin_Sentiment_MLPipeline/data/kill_switch_state.json).
- **Recovery Invariant:** State survives application restarts and power cycles without resetting to inactive.
- **Fail-Safe Design:** If state file read encounters an I/O or JSON corruption error, the system defaults to safe `ACTIVE` mode.

### Lifecycle & Behavior:
1. **Activation:**
   - Callable via `RiskGuard.activate_kill_switch(reason, triggered_by)`.
   - Immediately blocks all new incoming orders with violation code `KILL_SWITCH_ENGAGED`.
   - Records UTC timestamp, triggering agent/user, and justification into immutable event history.
2. **Non-Liquidation Policy:**
   - Activation blocks *new* trades. It does **not** force-liquidate existing holdings, avoiding market impact and fire-sale costs during operational distress.
3. **Deactivation:**
   - Requires explicit authorized invocation: `RiskGuard.deactivate_kill_switch(reason, approved_by)`.
   - Audited with clearance justification before trading is resumed.

### Audit Trail Verification:
Verified in automated test suite `tests/test_sprint10_suite.py::TestRiskGuardAndKillSwitch::test_kill_switch_activation_and_deactivation`.
