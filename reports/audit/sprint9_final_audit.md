# Sprint 9.1 — Final Quantitative Integrity Audit & Backtest Validation Report

**Audit Authority:** Senior Quantitative Researcher, ML Engineer & Financial Systems Auditor  
**Repository:** `Fin_Sentiment_MLPipeline`  
**Audit Standard:** Strict Walk-Forward Non-Lookahead, Out-of-Sample Separation, GIPS Alignment  
**Audit Date:** October 10, 2026  
**Final Audit Verdict:** **FAIL (CONDITIONAL ON REMEDIATION)**  

---

## 1. Executive Verdict & Core Determinations

Sprint 9 delivered a sophisticated multi-agent governance and adaptive intelligence pipeline. However, an independent forensic quantitative audit establishes that the headline financial and predictive claims cannot be accepted in their current form:

1. **The Claim of "+118.44% Paper Trading Return & +98.19 pp Alpha" is REJECTED (CRITICAL DEFECT):**  
   Simulated trade candidate selection in `evaluation/paper_trader.py:172` explicitly filtered candidate equities with `cand.target == 1` (a binary label indicating that 5-day forward price returns are strictly positive). When this future-peeking filter is removed, the clean walk-forward strategy return drops from **+118.44% to +11.32%**.
2. **The Claim of "+20.25% SPY Benchmark Return" is REJECTED (HIGH DEFECT):**  
   Lacking SPY index bars in the local database, `evaluation/paper_trader.py:116` substituted **Apple Inc. equity (`AAPL`)** as the market benchmark. Over the evaluation period (2025-10-01 to 2026-06-01), the actual S&P 500 Index (SPY) returned **+14.51%**, and an equal-weight universe portfolio returned **+22.51%**. The true alpha of the clean strategy against SPY is **$-3.19$ percentage points**, not $+98.19$ percentage points.
3. **The Claim of "87.5% Prediction Hit Rate Across 6,005 Observations" is MISLEADING (HIGH DEFECT):**  
   The 87.5% hit rate represents **7 correct predictions out of only 8 observations** ($N=8$, a $0.133\%$ selection rate) that crossed the 0.60 confidence threshold. The model's true recall across the 3,162 upward market moves was **0.22%**, overall accuracy was **47.44%** (underperforming the naive majority baseline of 52.66%), and ROC-AUC was **0.5062** (indistinguishable from a coin toss).
4. **The Claim of "+41.16% Capital Saved by Committee Vetoes" is FLAWED ARITHMETIC (MEDIUM DEFECT):**  
   `evaluation/committee_evaluator.py:198` calculated capital saved by summing unweighted percentage returns across 12 avoided loss trades. Crucially, the 9 rejected trades that subsequently gained value represented **$+46.74\%$ in missed gains** (including Intel at $+27.86\%$). The net counterfactual return of the committee vetoes was **$-5.58\%$**.
5. **Adaptive Committee Weights & Probability Calibration are MATHEMATICALLY SOUND BUT REVEAL CALIBRATION CONTAMINATION:**  
   The reported weights (Bull 33.3%, Bear 29.0%, Risk Officer 37.8%) sum to 100.1% purely due to a 1-decimal display rounding artifact (internal floating sum is exactly $1.0000$). However, probability calibration was fitted on data overlapping the training set ($[0.60, 0.80]$), distorting post-processing calibrators and favoring the raw uncalibrated model.

---

## 2. Audit Verification Matrix of Sprint 9 Claims

| Claim / Headline Metric | Reported Value | Audit Determination | Verified / Recomputed Value | Primary Root Cause & Evidence |
| :--- | :---: | :---: | :---: | :--- |
| **Out-of-Sample Observations** | 6,005 | **VERIFIED** | 6,005 | Test split indices $[24,020, 30,025]$ spanning 2025-09-23 to 2026-09-08. |
| **Prediction Hit Rate at Threshold**| 87.5% | **SUBSET ARTIFACT** | **87.50% ($7 / 8$ observations)** | Restricted to $N=8$ ($0.133\%$ selection rate); true recall is $0.22\%$. |
| **Overall ML Model Accuracy** | *Unreported* | **INDEPENDENT AUDIT** | **47.44%** | Fails naive 52.66% majority-class baseline. |
| **Model Discrimination (ROC-AUC)** | *Unreported* | **INDEPENDENT AUDIT** | **0.5062** | Statistically indistinguishable from random 0.50. |
| **Rank Information Coefficient**| *Unreported* | **INDEPENDENT AUDIT** | **$-0.0075$ ($t = -0.271$)** | Statistically insignificant ranking power. |
| **Calibration Champion** | Raw, ECE = 0.0355 | **QUALIFIED VERIFICATION**| Raw, ECE = 0.0355 | Low ECE driven by extreme probability compression near 0.50. |
| **Initial Paper Trading Capital** | ₹10,00,000 | **VERIFIED** | ₹10,00,000.00 | Exact initial cash balance. |
| **Final Paper Trading Capital** | ₹21,84,392.11 | **LEAKED ARTIFACT** | **₹11,13,205.33 (Clean)** | Stored run used future `cand.target == 1` filter. |
| **Paper Trading Cumulative Return** | +118.44% | **REJECTED** | **+11.32% (Clean)** | Dropped from +118.44% after removing lookahead bias. |
| **SPY Benchmark Return** | +20.25% | **REJECTED** | **+14.51% (True SPY)** | Stored run used AAPL stock as proxy benchmark. |
| **Alpha versus SPY** | +98.19 pp | **REJECTED** | **$-3.19$ percentage points** | Clean strategy underperformed true SPY benchmark. |
| **Equal-Weight Universe Return** | *Unreported* | **INDEPENDENT AUDIT** | **+22.51%** | 25-stock rebalanced universe return over identical period. |
| **Veto Loss-Prevention Accuracy**| 57.1% | **VERIFIED** | **57.14% ($12 / 21$)** | 12 of 21 vetoed setups had forward net return $< 0$. |
| **Capital Saved by Vetoes** | +41.16% | **REJECTED** | **$-5.58\%$ Net Economic Benefit** | Avoided losses (+41.16%) were outweighed by missed gains (+46.74%). |
| **Adaptive Committee Weights** | Bull 33.3%, Bear 29.0%, Risk 37.8% | **VERIFIED** | Bull 33.29%, Bear 28.95%, Risk 37.76% | Internal sum is $1.0000$; 100.1% is display rounding. |

---

## 3. Confirmed Defects & Severity Classification

### DEF-01 (CRITICAL) — Target Peeking in Paper Trading Simulation
- **File:** `evaluation/paper_trader.py:172`
- **Mechanism:** `qualified = [c for c in daily_cands if (c.target == 1 and ...)]`
- **Impact:** Simulator only bought assets guaranteed to gain over the forward 5 days. Artificial return inflation of $+107.12$ percentage points (+118.44% reported vs +11.32% clean).

### DEF-02 (HIGH) — Benchmark Proxy Substitution
- **File:** `evaluation/paper_trader.py:116`
- **Mechanism:** `spy_bars = session.query(MarketData).filter(MarketData.ticker == "AAPL").all()`
- **Impact:** Reported AAPL return (+20.25%) as SPY benchmark return, concealing true SPY return (+14.51%) and producing distorted alpha.

### DEF-03 (HIGH) — In-Sample Calibration Contamination & Threshold Peeking
- **File:** `evaluation/calibration.py:158, 191`
- **Mechanism:** Calibration split $[18,015, 24,020]$ overlapped the training period of `best_model.pkl` ($[0, 24,020]$), and `optimize_threshold()` fitted thresholds directly on `y_eval`.
- **Impact:** Artificially degraded Platt and Isotonic calibrators, and introduced test-set threshold optimization.

### DEF-04 (HIGH) — Denominator Concealment on Prediction Hit Rate
- **File:** `evaluation/prediction_evaluator.py:180`
- **Mechanism:** Headline reporting paired 87.5% hit rate with $N=6,005$ without disclosing that only $N=8$ observations were selected.
- **Impact:** Misled stakeholders regarding model signal frequency and operational capacity.

### DEF-05 (MEDIUM) — Flawed Capital Saved Formula
- **File:** `evaluation/committee_evaluator.py:198`
- **Mechanism:** `sum(v["loss_avoided"] for v in vetoes)` summed unweighted asset returns and ignored missed gains (+46.74%).
- **Impact:** Created false impression of positive committee risk attribution when net counterfactual alpha was $-5.58\%$.

### DEF-06 (MEDIUM) — Unpopulated Outcome Resolution Tables
- **File:** `evaluation/outcome_tracker.py`
- **Mechanism:** `prediction_outcomes` and `opportunity_outcomes` tables had 0 persisted rows.
- **Impact:** Hampered automated continuous feedback loops until batch resolution is run.

---

## 4. Comprehensive Performance Reconciliation Table

| Metric | Stored Leaked Run | Clean Walk-Forward (No Leakage) | True SPY Benchmark | Equal-Weight 25 Universe |
| :--- | :---: | :---: | :---: | :---: |
| **Initial Capital** | ₹10,00,000.00 | ₹10,00,000.00 | N/A | N/A |
| **Final Capital** | ₹21,84,392.11 | **₹11,13,205.33** | N/A | N/A |
| **Cumulative Return** | +118.44% | **+11.32%** | **+14.51%** | **+22.51%** |
| **Annualized Return (CAGR)**| +223.63% | **+17.38%** | **+22.52%** | **+35.80%** |
| **Annualized Volatility** | 234.98% | **22.40%** | **13.03%** | **14.18%** |
| **Sharpe Ratio** | 1.48 | **0.62** | **1.25** | **1.42** |
| **Sortino Ratio** | 2.63 | **0.95** | **1.84** | **2.10** |
| **Calmar Ratio** | 2.87 | **0.91** | **1.64** | **2.01** |
| **Max Drawdown** | 19.51% | **12.45%** | **8.85%** | **11.20%** |
| **Alpha vs SPY** | +98.19 pp | **$-3.19$ pp** | 0.00 pp | +8.00 pp |
| **Profit Factor** | 4.58 | **1.24** | N/A | N/A |
| **Win Rate** | 72.27% | **52.11%** | N/A | N/A |
| **Closed Trades** | 768 | **213** | N/A | N/A |
| **Round-Trip Friction** | 20 bps | **20 bps** | N/A | N/A |

---

## 5. Surgical Remediation Plan

To transition Sprint 9 from **FAIL** to **PASS WITH LIMITATIONS**, the following surgical remediations must be executed:

1. **Remediate `evaluation/paper_trader.py`:**
   - Remove `c.target == 1` from candidate selection on line 173.
   - Filter trade orders strictly using decision-time features:
     ```python
     qualified = [
         c for c in daily_cands 
         if (c.rsi is None or c.rsi < 70) 
         and (c.avg_sentiment is None or c.avg_sentiment >= 0.0)
     ]
     ```
   - Ingest true SPY prices or compute benchmark returns using `TrainingFeature.spy_return_1d`.
2. **Remediate `evaluation/committee_evaluator.py`:**
   - Fix capital saved formula to report dual-sided metrics: Avoided Losses, Missed Gains, and Net Counterfactual Benefit.
3. **Remediate `evaluation/calibration.py`:**
   - Re-partition train/cal/eval splits so `best_model.pkl` is fitted on $[0, 0.60]$, calibrators on $[0.60, 0.80]$, and evaluation on $[0.80, 1.00]$.
   - Optimize decision thresholds on `(y_cal, p_cal)`, never on `y_eval`.

---

## 6. Recommendation on Sprint 10

### Can Sprint 10 Proceed?
**YES, CONDITIONAL ON REMEDIATION.**  
The core architecture (multi-agent committee, FAISS historical analog engine, evidence prosecutor, adaptive weight formulation, transaction cost model) is conceptually robust and institutionally sound. However, **all promotional references to "+118.44% return", "+98.19 pp alpha", and "87.5% hit rate" must be permanently retracted and replaced with the verified baseline (+11.32% return, $-3.19$ pp alpha, 52.1% win rate)**. 

Sprint 10 (Live Execution & Automated Risk Controls) can safely proceed once the surgical fixes are committed and verified against `scripts/audit_sprint9_integrity.py`.
