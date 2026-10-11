# SPRINT 9.1 — INITIAL AUDIT REPORT: REPOSITORY & BASELINE AUDIT
**Date of Audit:** 2026-10-09  
**Auditor Role:** Senior Quantitative Researcher & Financial Systems Auditor  
**Repository:** `Fin_Sentiment_MLPipeline`  
**Commit Audited:** `804524d` ("Autonomous Evaluation & Adaptive Intelligence implemented")  

---

## 1. Executive Summary of Phase 0 Audit

An exhaustive forensic inspection of the codebase, database artifacts, dataset splits, evaluation engines, and generated markdown reports was conducted. 

### Key High-Level Findings:
1. **Source of 6,005 Observations:** Sourced directly from `data/training_dataset.csv` (30,025 rows, 25 tickers) via a chronological 80/20 train/test temporal split (`split_ratio=0.80`). The test set spans `2025-09-23` to `2026-09-08`.
2. **The 87.5% Hit Rate Reality:** The reported "87.5% hit rate at threshold" is based on **exactly $N = 8$ positive predictions** out of 6,005 test observations (0.13% selection rate). The model's overall test set accuracy is **47.4%**, with an ROC-AUC of **0.5062**, F1 score of **0.0044**, and Spearman Rank Information Coefficient (IC) of **-0.0075**.
3. **Critical Target Leakage in Paper Trading:** `evaluation/paper_trader.py` filters prospective trade candidates using `cand.target == 1` directly from `TrainingFeature`. Because `target` is the ground-truth binary label defined as `future_return_5d > 0`, the paper trader literally inspects future outcomes before entering positions.
4. **Benchmark Substitution:** `evaluation/paper_trader.py` lacks SPY price bars in `MarketData` and quietly substitutes **AAPL** as the SPY proxy benchmark. The reported +20.25% "SPY return" is actually the individual stock return of Apple Inc.
5. **Empty Database Tables:** `prediction_outcomes` and `opportunity_outcomes` contain **0 rows** in the active SQLite database (`data/fin_sentiment.db`), as their batch resolution methods were defined but never executed in the validation pipeline.
6. **Synthetic Historical Agent Cohorts:** Because historical `committee_votes` were only recorded for recent live demos (`2026-09-14` to `2026-09-20`), `AgentEvaluator` created synthetic agent decisions via heuristic formula proxies to evaluate agents on dates where forward market data was available.
7. **Flawed Capital Saved Arithmetic:** The reported "+41.16% capital saved by vetoes" is the arithmetic sum of unweighted asset percentage drawdowns across 12 trades, rather than a portfolio-level dollar or basis point saving.

---

## 2. Inventory of Audited Components & Persisted Records

### 2.1 Database Tables & Row Counts (`data/fin_sentiment.db`)
| Table Name | Active Row Count | Primary Key | Description & Population Mechanism |
| :--- | :---: | :---: | :--- |
| `prediction_outcomes` | **0** | UUID String | **EMPTY.** Method `resolve_prediction_outcomes()` was never invoked. |
| `agent_decision_records` | **684** | UUID String | Populated by `AgentEvaluator.sync_agent_decision_records()` (synthetic + demo). |
| `agent_evaluation_records` | **18** | UUID String | Populated by `AgentEvaluator.evaluate_all_agents()` across 6 agents. |
| `committee_evaluation_records` | **3** | UUID String | Populated by `CommitteeEvaluator.evaluate_committee_counterfactuals()`. |
| `opportunity_outcomes` | **0** | UUID String | **EMPTY.** Method `resolve_opportunity_outcomes()` was never invoked. |
| `paper_trading_positions` | **768** | UUID String | Populated by `PaperTrader.run_walk_forward()`. |
| `paper_trading_snapshots` | **668** | UUID String | Populated by `PaperTrader.run_walk_forward()` daily equity curves. |
| `model_calibration_records` | **3** | UUID String | Populated by `ProbabilityCalibrator.run_temporal_calibration_experiment()`. |
| `evaluation_snapshots` | **1** | UUID String | Populated by `MasterPerformanceReporter.run_full_evaluation()`. |
| `committee_decisions` | **65** | UUID String | Real Sprint 6/7/8 committee governance decision history. |
| `committee_votes` | **259** | UUID String | Real individual committee member vote records. |
| `training_features` | **30,025** | UUID String | Engineered feature store across 25 universe tickers. |
| `market_data` | **31,375** | UUID String | Raw daily price bars from `2021-09-15` to `2026-09-15`. |

---

### 2.2 Entry Points & Core Script Map
| Script / Entry Point | Purpose | Output / Artifact |
| :--- | :--- | :--- |
| `evaluation/metrics.py` | Statistical & financial formulas | Core calculation library |
| `evaluation/outcome_tracker.py` | Multi-horizon forward return resolution | Session lookups without look-ahead |
| `evaluation/prediction_evaluator.py` | Out-of-sample frozen model evaluation | `reports/evaluation_reports/sprint9_baseline.md` |
| `evaluation/agent_evaluator.py` | Independent agent performance attribution | `reports/evaluation_reports/agent_performance.md` |
| `evaluation/committee_evaluator.py` | Counterfactuals & veto loss prevention | `reports/evaluation_reports/committee_performance.md` |
| `evaluation/calibration.py` | Platt & Isotonic calibration | Persists `ModelCalibrationRecord` |
| `evaluation/paper_trader.py` | Walk-forward portfolio simulation | `reports/evaluation_reports/paper_trading_performance.md` |
| `evaluation/adaptive_weights.py` | Dynamic committee voting weight engine | Returns updated agent weights & walk-forward test |
| `evaluation/adaptive_opportunities.py` | Rolling IC opportunity factor weights | Returns updated factor weights & walk-forward test |
| `evaluation/performance_report.py` | Master orchestrator & DB snapshot saver | `reports/evaluation_reports/sprint9_evaluation_report.md` |
| `scripts/validate_sprint9_evaluation.py` | End-to-end Sprint 9 test suite | 11 verification gates |

---

## 3. Forensic Trace of the Headline Claims

### Claim 1: "Out-of-sample prediction observations: 6,005"
- **Verification Status:** **CONFIRMED (SOURCE IDENTIFIED)**
- **Evidence:** `data/training_dataset.csv` contains 30,025 observations across 25 tickers (`AAPL`, `AMD`, `AMZN`, `BAC`, `CAT`, `CRM`, `CVX`, `GOOGL`, `GS`, `HD`, `HON`, `INTC`, `JNJ`, `JPM`, `META`, `MSFT`, `NFLX`, `NVDA`, `ORCL`, `PFE`, `TSLA`, `UNH`, `V`, `WMT`, `XOM`) covering `2021-11-23` to `2026-09-08`.
- In `models/feature_store.py::temporal_train_test_split(df, split_ratio=0.80)`:
  $$\text{split\_idx} = \lfloor 30,025 \times 0.80 \rfloor = 24,020$$
  $$\text{Train} = 24,020 \text{ rows } (2021-11-23 \text{ to } 2025-09-23)$$
  $$\text{Test} = 6,005 \text{ rows } (2025-09-23 \text{ to } 2026-09-08)$$
- **Limitation:** The 6,005 observations represent cross-sectional daily bars across 25 tickers over ~240 trading days with 5-day overlapping forward return targets. They are **not 6,005 independent experiments**.

---

### Claim 2: "Prediction hit rate at threshold: 87.5%"
- **Verification Status:** **MISLEADING DENOMINATOR ($N=8$)**
- **Evidence:**
  At threshold $\tau = 0.60$:
  - Predictions with $\hat{p} \ge 0.60$: **8** (out of 6,005, or **0.13%** of all test observations).
  - True positive outcomes among those 8: **7** ($7 / 8 = 87.5\%$).
  - Mean 5-day return of those 8 setups: $+5.00\%$.
- **Audit Defect:** The report failed to disclose that $N = 8$. Readers were left to assume 87.5% applied across a meaningful fraction of the 6,005 test observations.
- **Unconditional Baseline:** The majority-class baseline (unconditional positive rate in the test set) is **52.66%**. The model's overall accuracy across all 6,005 observations is **47.4%** (worse than random guessing).

---

### Claim 3: "Paper-trading capital: ₹10,00,000 -> ₹21,84,392.11 (+118.44% return, +98.19% alpha)"
- **Verification Status:** **INVALIDATED (CONFIRMED TARGET LEAKAGE & BENCHMARK SUBSTITUTION)**
- **Evidence:**
  1. **Direct Target Leakage in Trade Generation (`evaluation/paper_trader.py:L172`):**
     ```python
     daily_cands = candidates_by_date[cur_date]
     qualified = [c for c in daily_cands if (c.target == 1 and ...)]
     ```
     `TrainingFeature.target` is the ground-truth future target defined as $1$ if $\text{future\_return\_5d} > 0$. The simulator filtered candidates at entry time using the future outcome label, ensuring almost every entered trade was pre-screened to be a winner.
  2. **AAPL Substituted for SPY (`evaluation/paper_trader.py:L88`):**
     ```python
     spy_bars = session.query(MarketData).filter(MarketData.ticker == "AAPL").all() # Proxy benchmark
     ```
     Because `SPY` bars do not exist in the 25-stock `market_data` table, the code fetched `AAPL` prices and reported Apple's $+20.25\%$ return as "Benchmark (SPY) Return", calculating $+98.19\%$ alpha against Apple stock rather than the S&P 500 index.
  3. **Arithmetic Reconciliation:**
     $$\frac{₹21,84,392.11 - ₹10,00,000}{₹10,00,000} = +118.439\% \approx +118.44\%$$
     The arithmetic of the return is mechanically correct given the equity curve, but the equity curve itself is mathematically invalid due to forward target leakage.

---

### Claim 4: "Committee veto loss-prevention accuracy: 57.1% and Capital saved: +41.16%"
- **Verification Status:** **PARTIALLY VALIDATED WITH FLAWED AGGREGATION**
- **Evidence:**
  - Evaluated on $N = 21$ vetoed setups:
    - Net return $< 0$ after 10 bps friction: **12** ($12 / 21 = 57.14\%$).
    - Net return $\ge 0$ (missed profits): **9** ($9 / 21 = 42.86\%$).
  - **Capital Saved Flaw (`evaluation/committee_evaluator.py:L138`):**
    ```python
    total_capital_saved = sum(v["loss_avoided"] for v in vetoes)
    ```
    The code simply summed unweighted percentage losses (e.g., $3.2\% + 4.1\% + 2.8\% + \dots = 41.16\%$). This is an invalid aggregation. Avoiding a 3% loss on a 10% position saves 0.3% of portfolio equity, not 3%.

---

### Claim 5: "Adaptive committee weights: Bull 33.3%, Bear 29.0%, Risk Officer 37.8% (Sum = 100.1%)"
- **Verification Status:** **CONFIRMED AS ROUNDING DISPLAY ARTIFACT**
- **Evidence:**
  The underlying floating-point weights computed by `AdaptiveCommitteeWeights._normalize_and_bound_weights()` were:
  - Bull Analyst: `0.3329` ($33.29\%$)
  - Bear Analyst: `0.2895` ($28.95\%$)
  - Risk Officer: `0.3776` ($37.76\%$)
  - Sum: $0.3329 + 0.2895 + 0.3776 = 1.0000$ ($100.00\%$).
  Formatting each to 1 decimal place produced $33.3\% + 29.0\% + 37.8\% = 100.1\%$. The internal weights strictly sum to 1.0.
  However, Portfolio Manager, Evidence Prosecutor, and CIO were excluded from the baseline governance weights.

---

## 4. Summary of Identified Defects to Audit in Detail

| Defect ID | Description | Severity | Affected File |
| :--- | :--- | :---: | :--- |
| **DEF-01** | Forward target label (`c.target == 1`) used as trade entry filter in paper trading | **CRITICAL** | `evaluation/paper_trader.py` |
| **DEF-02** | AAPL stock price substituted as proxy for SPY benchmark return | **HIGH** | `evaluation/paper_trader.py` |
| **DEF-03** | In-sample training overlap in probability calibration fitting | **HIGH** | `evaluation/calibration.py` |
| **DEF-04** | Headline 87.5% hit rate reported without disclosing tiny sample size ($N=8$) | **HIGH** | `reports/evaluation_reports/sprint9_baseline.md` |
| **DEF-05** | Summation of unweighted percentage returns to claim "41.16% capital saved" | **MEDIUM** | `evaluation/committee_evaluator.py` |
| **DEF-06** | `prediction_outcomes` and `opportunity_outcomes` tables left completely empty | **MEDIUM** | `evaluation/outcome_tracker.py` |
| **DEF-07** | Synthetic proxy heuristics used instead of executing actual agent classes | **MEDIUM** | `evaluation/agent_evaluator.py` |

---
*End of Phase 0 Initial Audit. Proceeding to Phase 1: Temporal Leakage Audit.*
