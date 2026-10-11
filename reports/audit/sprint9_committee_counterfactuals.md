# Sprint 9.1 — Committee & Veto Counterfactual Audit Report

**Audit Lead:** Senior Quantitative Researcher & Financial Systems Auditor  
**Repository:** `Fin_Sentiment_MLPipeline`  
**Module Audited:** `evaluation/committee_evaluator.py`  
**Dataset:** 60 Reconstructed Investment Committee Cohorts (2025-10-01 to 2026-06-01)  
**Classification:** Institutional Quantitative Audit  

---

## 1. Executive Summary

Sprint 9 reported two headline governance achievements:
1. **"Committee veto loss-prevention accuracy: 57.1%"**
2. **"Capital saved by vetoes: +41.16%"**

### Forensic Finding: **FLAWED CAPITAL SAVED METRIC & NET NEGATIVE VETO ATTRIBUTION**

1. **Veto Accuracy ($57.1\%$) Verified Mathematically:**  
   The Committee evaluated 21 vetoes where the Bull/ML signal proposed an order but the Risk Officer (CRO) or Evidence Prosecutor exercised a veto. Of these, 12 positions suffered negative 5-day net returns ($12 / 21 = 57.14\%$).
2. **Flawed "+41.16% Capital Saved" Arithmetic:**  
   In `evaluation/committee_evaluator.py:198`, the metric `total_capital_saved` was calculated by **summing unweighted percentage returns** across the 12 avoided loss trades ($9.56\% + 0.85\% + 1.88\% + \dots = 41.16\%$). This does not reflect portfolio capital saved; it is an arithmetic summation of asset price changes.
3. **Severe Concealment of Missed Gains (+46.74%):**  
   For the remaining 9 vetoed candidates where the veto was wrong, the underlying equities generated substantial positive returns, including **Intel (+27.86% in 5 days)** and **Caterpillar (+5.44%)**.
   - Total Avoided Losses: **$+41.16\%$**
   - Total Missed Gains: **$-46.74\%$**
   - **Net Counterfactual Economic Impact of Vetoes:** **$-5.58\%$ (Net Detriment)**.
   The committee's vetoes actually destroyed net potential alpha compared to the unconstrained ML model.

---

## 2. Complete Ledger of 21 Committee Vetoes

All 21 vetoed candidate setups audited with net returns (after 10 bps friction):

| Date | Ticker | Raw Return 5d | Net Return 5d | Veto Prevented Loss? | Counterfactual Avoided Loss | Missed Opportunity Gain |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 2025-10-01 | AAPL | +1.02% | +0.92% | False | 0.00% | +0.92% |
| 2025-10-03 | PFE | $-9.46\%$ | **$-9.56\%$** | **True** | **+9.56%** | 0.00% |
| 2025-10-08 | INTC | $-0.75\%$ | **$-0.85\%$** | **True** | **+0.85%** | 0.00% |
| 2025-10-13 | CAT | +5.54% | +5.44% | False | 0.00% | +5.44% |
| 2025-10-23 | JNJ | $-1.78\%$ | **$-1.88\%$** | **True** | **+1.88%** | 0.00% |
| 2025-10-31 | GOOGL | $-0.84\%$ | **$-0.94\%$** | **True** | **+0.94%** | 0.00% |
| 2025-11-07 | JPM | $-3.37\%$ | **$-3.47\%$** | **True** | **+3.47%** | 0.00% |
| 2025-11-28 | GS | +3.97% | +3.87% | False | 0.00% | +3.87% |
| 2025-12-05 | WMT | +1.59% | +1.49% | False | 0.00% | +1.49% |
| 2026-01-06 | BAC | $-4.73\%$ | **$-4.83\%$** | **True** | **+4.83%** | 0.00% |
| 2026-01-16 | INTC | $-9.52\%$ | **$-9.62\%$** | **True** | **+9.62%** | 0.00% |
| 2026-01-22 | CAT | +2.60% | +2.50% | False | 0.00% | +2.50% |
| 2026-01-27 | AMZN | $-2.48\%$ | **$-2.58\%$** | **True** | **+2.58%** | 0.00% |
| 2026-02-03 | JNJ | +2.25% | +2.15% | False | 0.00% | +2.15% |
| 2026-02-24 | CVX | +1.85% | +1.75% | False | 0.00% | +1.75% |
| 2026-04-16 | BAC | $-1.94\%$ | **$-2.04\%$** | **True** | **+2.04%** | 0.00% |
| 2026-04-28 | INTC | **+27.96%** | **+27.86%** | False | 0.00% | **+27.86%** |
| 2026-05-01 | CAT | +0.87% | +0.77% | False | 0.00% | +0.77% |
| 2026-05-06 | AMZN | $-1.77\%$ | **$-1.87\%$** | **True** | **+1.87%** | 0.00% |
| 2026-05-08 | TSLA | $-1.43\%$ | **$-1.53\%$** | **True** | **+1.53%** | 0.00% |
| 2026-05-21 | GOOGL | $-1.89\%$ | **$-1.99\%$** | **True** | **+1.99%** | 0.00% |
| **TOTAL** | — | — | — | **12 / 21 (57.1%)** | **+41.16%** | **+46.74%** |

$$\text{Net Economic Benefit} = \sum \text{Avoided Losses} - \sum \text{Missed Gains} = 41.16\% - 46.74\% = \mathbf{-5.58\%}$$

---

## 3. Counterfactual Strategy Comparison

Comparison across identical dates, candidate pools, and 10 bps transaction friction:

| Strategy | Trades | Win Rate | Annualized Return | Sharpe Ratio | Max Drawdown | Alpha vs SPY |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Committee Governance** | 24 | 54.2% | +8.42% | 0.58 | 11.2% | $-6.09$ pp |
| **ML-Only Strategy** | 35 | 51.4% | +14.00% | 0.74 | 14.8% | $-0.51$ pp |
| **Opportunity-Only Strategy**| 35 | 51.4% | +14.00% | 0.74 | 14.8% | $-0.51$ pp |
| **Equal-Weight Consensus** | 42 | 52.4% | +11.85% | 0.69 | 12.6% | $-2.66$ pp |
| **SPY Buy & Hold** | Buy & Hold | N/A | +14.51% | 1.25 | 8.85% | 0.00 pp |
| **Equal-Weight 25 Universe**| Rebalanced | N/A | +22.51% | 1.42 | 11.20% | +8.00 pp |

### Key Institutional Takeaway
While the Committee successfully suppressed downside volatility (reducing Max Drawdown from 14.8% to 11.2%), it did so at the expense of annualized returns (+8.42% vs +14.00% for ML-only). By vetoing high-beta breakout trades like INTC (+27.86%), the committee reduced risk but created negative net value add.

---

## 4. Audit Recommendations

1. **Replace Flawed Capital Saved Formula:**  
   Discontinue summing percentage returns across trades. Instead, report:
   - Net portfolio cash impact: $\sum w_i \cdot \text{Loss Avoided}_i - \sum w_i \cdot \text{Missed Gain}_i$.
2. **Dual-Sided Veto Reporting:**  
   Whenever veto loss-prevention accuracy is presented, the report must disclose both the avoided losses and the missed opportunity costs.
