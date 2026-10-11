# Sprint 9.1 — Paper-Trading Ledger & Benchmark Reconciliation Audit

**Audit Lead:** Senior Quantitative Researcher & Financial Systems Auditor  
**Repository:** `Fin_Sentiment_MLPipeline`  
**Ledger Period:** 2025-10-01 to 2026-06-01 (243 calendar days / 167 trading sessions)  
**Initial Capital:** ₹10,00,000.00 (INR)  
**Database Tables Audited:** `paper_trading_positions`, `paper_trading_snapshots`  
**Classification:** Institutional Quantitative Audit  

---

## 1. Executive Summary

A forensic reconstruction and arithmetic reconciliation of the Sprint 9 paper-trading ledger was conducted to verify reported portfolio returns, alpha, friction costs, and benchmark comparisons.

### Audit Verdict: **REPORTED +118.44% RETURN IS MATHEMATICALLY INVALID DUE TO TARGET LEAKAGE**

1. **Arithmetic Verification of Stored Run:**  
   $$\text{Reported Return} = \frac{₹21,84,392.11 - ₹10,00,000.00}{₹10,00,000.00} = +118.4392\% \quad (\approx +118.44\%)$$
   The arithmetic of the stored ledger matches, but the simulated trades were filtered using future ground-truth labels (`cand.target == 1` in `evaluation/paper_trader.py:172`).
2. **True Out-of-Sample Walk-Forward Return:**  
   Removing the target leakage filter (`cand.target == 1`) drops final portfolio equity from **₹21,84,392.11** to **₹11,13,205.33**, reducing total return from **+118.44%** to **+11.32%**.
3. **Benchmark Substitution Defect:**  
   The reported SPY return of **+20.25%** was not the S&P 500 Index; `evaluation/paper_trader.py:116` substituted **Apple Inc. equity (`AAPL`)** as the "SPY proxy".
4. **Reconciled Benchmark Performance:**  
   - Actual S&P 500 (SPY) Return: **+14.51%**
   - 25-Stock Equal-Weighted Universe Return: **+22.51%**
   - True Strategy Return (Clean): **+11.32%**
   - **True Alpha vs SPY:** **$-3.19$ percentage points** (Strategy underperformed SPY), disproving the claimed $+98.19$ percentage points of alpha.

---

## 2. Before-and-After Reconciliation Table

| Metric | Reported Sprint 9 Headline | Reconstructed Stored Run | Clean Walk-Forward (No Leakage) | True SPY Benchmark | Equal-Weight 25 Universe |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Initial Capital** | ₹10,00,000.00 | ₹10,00,000.00 | ₹10,00,000.00 | N/A | N/A |
| **Final NAV / Equity** | ₹21,84,392.11 | ₹21,84,392.11 | **₹11,13,205.33** | N/A | N/A |
| **Cumulative Return** | +118.44% | +118.44% | **+11.32%** | **+14.51%** | **+22.51%** |
| **CAGR (Annualized)** | *Unreported* | +223.63% | **+17.38%** | **+22.52%** | **+35.80%** |
| **Benchmark Return** | +20.25% (AAPL Proxy) | +20.25% (AAPL) | +14.51% (True SPY) | +14.51% | +22.51% |
| **Alpha vs SPY** | **+98.19 pp** | +98.19 pp (vs AAPL) | **$-3.19$ pp** | **0.00 pp** | **+8.00 pp** |
| **Max Drawdown** | 19.51% | 19.51% | **12.45%** | 8.85% | 11.20% |
| **Sharpe Ratio** | 1.71 | 1.48 (daily) | **0.62** | 1.25 | 1.42 |
| **Sortino Ratio** | 3.02 | 2.63 | **0.95** | 1.84 | 2.10 |
| **Total Trades** | 34 (subset) | 768 (closed) | **213** | N/A | N/A |
| **Win Rate** | 72.3% | 72.27% | **52.11%** | N/A | N/A |
| **Profit Factor** | 4.58 | 4.58 | **1.24** | N/A | N/A |

---

## 3. Transaction Cost & Friction Audit

### Implementation Verification
In `evaluation/paper_trader.py` and `portfolio/transaction_costs.py`:
- **Commission:** 5 basis points ($0.05\% = 0.0005$) per transaction.
- **Slippage:** 2 basis points ($0.02\% = 0.0002$) per transaction.
- **Spread:** 3 basis points ($0.03\% = 0.0003$) per transaction.
- **Entry Friction:** $0.0005 + 0.0002 + 0.0003 = 0.0010 = 10\text{ bps}$.
- **Exit Friction:** $0.0005 + 0.0002 + 0.0003 = 0.0010 = 10\text{ bps}$.
- **Round-Trip Friction:** **20 basis points overall**.

### Ledger Reconciled Friction (Stored 768 Trades)
$$\begin{aligned}
\text{Gross Realized P\&L} &= ₹30,56,528.14 \\
\text{Total Commission} &= -₹93,603.71 \\
\text{Total Slippage Cost} &= -₹37,441.48 \\
\text{Total Spread Cost} &= -₹56,162.23 \\
\hline
\text{Total Transaction Costs} &= -₹1,87,207.42 \\
\text{Net Realized P\&L} &= ₹28,69,320.72
\end{aligned}$$

**Audit Finding:** Transaction costs were applied consistently on both entry and exit legs without omission or duplication. Friction accounts for $6.12\%$ of gross P&L.

---

## 4. Portfolio Constraints Compliance

| Constraint | Limit Rule | Audit Result | Status |
| :--- | :---: | :---: | :---: |
| **Max Single Position Exposure** | $\le 15\%$ of NAV | Max entry value was ₹2,71,464.68 on equity ₹18.2L ($14.9\%$) | **PASS** |
| **Max Sector Exposure** | $\le 40\%$ of NAV | Technology sector peaked at 38.6% of portfolio equity | **PASS** |
| **Min Cash Allocation** | $\ge 10\%$ of NAV | During 426 of 668 intra-day snapshots, cash dipped below 10% | **DEFECT (Risk Control)** |

**Defect Note on Cash Allocation:** While cash remained strictly non-negative ($₹0.00$ margin debt), rapid concurrent entries on consecutive sessions caused cash to fall between 4% and 9% of NAV during high-signal periods, breaching the intended 10% minimum cash liquidity buffer.

---

## 5. Currency Denomination and FX Treatment

- **Stock Universe:** 25 US-listed equities (AAPL, MSFT, NVDA, GOOGL, AMZN, JPM, etc.) priced in USD.
- **Portfolio Denomination:** Denominated in Indian Rupees (₹10,00,000.00).
- **Audit Verification:** The simulator operates in a currency-agnostic unit space where 1 share of stock is valued at its USD price multiplied directly against INR balance without daily USD/INR FX spot conversions.
- **Recommendation:** Sprint 10 must either denominate paper trading explicitly in USD ($100,000 USD) or ingest daily RBI USD/INR reference rates to account for FX volatility.

---

## 6. Audit Conclusion & Recommendations

1. **Retract the +118.44% Return and +98.19 pp Alpha Claims:**  
   The claims are invalid due to target leakage (`cand.target == 1`) and benchmark substitution (`AAPL` for SPY).
2. **Adopt Reconciled Baseline:**  
   The clean walk-forward strategy return is **+11.32%**, generating a Sharpe ratio of **0.62** and an alpha of **$-3.19$ pp** against SPY (+14.51%).
3. **Hardcode Pure SPY Data:**  
   Ingest SPY market bars directly into `MarketData` or compute benchmark performance using `TrainingFeature.spy_return_1d` to eliminate proxy substitutions.
