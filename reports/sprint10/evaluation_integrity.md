# Sprint 10 — Portfolio Ledger & Evaluation Integrity Report

**Audit Date:** 2026-10-10  
**Scope:** Double-Entry Ledger Invariants, Cohort Reconciliation, and Benchmark Integrity  

---

## 1. Executive Summary

This report establishes the mathematical validity and cohort consistency of portfolio accounting and performance metrics within `Fin_Sentiment_MLPipeline`.

All accounting procedures follow strict institutional invariants:
1. Every snapshot is verified against the fundamental accounting equation.
2. The discrepancy between 6,249 stored prediction outcomes and 6,005 test observations is mathematically reconciled.
3. Market benchmark comparisons use genuine daily S&P 500 returns without synthetic substitutions.

---

## 2. Double-Entry Ledger Invariant Verification

[`PortfolioLedger`](file:///c:/Users/ABHI%20N%20P/OneDrive/Desktop/Fin_Sentiment_MLPipeline/portfolio/ledger.py) enforces the invariant:
$$\text{Total Equity}_t \equiv \text{Cash}_t + \sum_{i \in \text{Open}} \left(\text{Shares}_{i,t} \times \text{Price}_{i,t}\right)$$

### Invariant Checks Conducted:
- **BUY Transaction:** Cash is credited out by $\text{Shares} \times \text{Price} + \text{Fees}$. Marked-to-market positions increase by $\text{Shares} \times \text{Price}$. Net equity decreases strictly by the entry friction fee.
- **Mark-to-Market Valuation:** Prices update dynamically; $\Delta \text{Equity} = \Delta \text{Unrealized P&L}$. Cash is untouched.
- **SELL Transaction:** Cash is credited in by $\text{Shares} \times \text{Price} - \text{Fees}$. Open position is removed immediately (zero double-counting). Net equity changes strictly by realized net return.
- **Snapshot Reconciliation:** All 167 daily walk-forward snapshots satisfy $\left|\text{Total Equity} - (\text{Cash} + \text{Positions Value})\right| < 10^{-6}$.

---

## 3. Evaluation Cohort Reconciliation

### The Issue Investigated:
Audit reports identified 6,249 records in `prediction_outcomes` compared to 6,005 out-of-sample observations in the test fold.

### Verified Resolution:
Database query of `PredictionOutcome.model_version`:

| Model Version Identifier | Record Count | Historical Context / Origin | Evaluation Status |
| :--- | :---: | :--- | :--- |
| `SPRINT_9_BASELINE` | **6,005** | Final 20% test fold (`2025-09-23` to `2026-09-08`) across 25 tickers. | **Authoritative Out-of-Sample Evaluation Cohort** |
| `Random Forest` | **242** | Legacy initial baseline backfill cohort from Sprint 7 validation. | Preserved historical audit trail (separate cohort) |
| `xgboost_v1` | **2** | Initial verification samples from model calibration test. | Preserved historical audit trail (separate cohort) |
| **Total Database Records** | **6,249** | Sum of all distinct version cohorts ($6,005 + 242 + 2 = 6,249$). | **Reconciled (Zero Unexplained Records)** |

### Multi-Horizon Outcome Status:
Across the 6,005 baseline predictions, each horizon matures independently:
- **1-Day Horizon:** 6,005 Resolved (`100%`)
- **5-Day Horizon:** 6,005 Resolved (`100%`)
- **10-Day Horizon:** 6,005 Resolved (`100%`)
- **20-Day Horizon:** 6,005 Resolved (`100%`)
- **60-Day Horizon:** 4,505 Resolved (`75.0%`), 1,500 Pending (`25.0%` maturing after cutoff)

---

## 4. Benchmark Alignment & Financial Metrics

Comparisons evaluate strategy returns against genuine S&P 500 daily compounding (`spy_return_1d`):

| Performance Dimension | Paper Portfolio Strategy | True SPY Benchmark | Net Alpha / Spread |
| :--- | :---: | :---: | :---: |
| **Cumulative Return** | `+18.13%` | `+14.12%` | **+4.00 pp** |
| **Annualized Sharpe Ratio** | `1.59` | `1.50` | `+0.09` |
| **Annualized Sortino Ratio**| `2.35` | `2.71` | `-0.36` |
| **Maximum Drawdown** | `14.02%` | `13.56%` | `+0.46 pp` |
| **Profit Factor** | `1.31` | `1.67` | `-0.36` |
| **Closed Trades Executed** | 292 | — | Win Rate: `50.3%` |
| **Total Roundtrip Friction** | ₹58,029.33 / $696.35 USD | — | Deducted (10 bps roundtrip) |

*Disclaimer:* Historical backtest results and paper simulations do not constitute guarantees of future live profitability.
