# Sprint 9.1 — Temporal Leakage and Data Integrity Audit

**Audit Lead:** Senior Quantitative Researcher & Financial Systems Auditor  
**Repository:** `Fin_Sentiment_MLPipeline`  
**Date of Audit:** October 10, 2026  
**Audit Standard:** Strict Non-Lookahead, Walk-Forward Causality, Out-of-Sample Separation  
**Classification:** Institutional Quantitative Audit  

---

## 1. Executive Summary

A forensic temporal audit of `Fin_Sentiment_MLPipeline` was conducted across feature generation, training splits, probability calibration, agent decision generation, opportunity ranking, and portfolio simulation.

### Critical Verdict: **CRITICAL LEAKAGE DETECTED**

While the foundational feature engineering pipeline in `models/feature_engineer.py` adheres to strict backward-lagging (`.shift(1)` on technicals and per-date cross-sectional ranking), **critical target leakage was discovered in the paper trading simulation engine (`evaluation/paper_trader.py`)**. Specifically, simulated order generation explicitly filtered trade candidates using future ground-truth labels (`cand.target == 1`), peeking directly into 5-day forward price returns. Furthermore, probability calibration was fitted on data overlapping the model's training period.

---

## 2. Forensic Trace of Pipeline Components

### 2.1 Feature Engineering (`models/feature_engineer.py`)
- **Status:** **PASS (Temporally Valid)**
- **Audit Findings:**
  1. Price and volume inputs: Rolling technical indicators (RSI-14, MACD, Bollinger Bands, ATR) are constructed using past prices. Return lags (`ret_1d_lag1`, `ret_5d_lag1`, `ret_20d_lag1`) use explicit `.shift(1)` offsets relative to feature calculation time $T$.
  2. Cross-sectional features: Metrics such as `cross_sectional_mom_z` and `mom_rank_sector` group by `date` at timestamp $T$. No future dates are pooled into the normalization transforms.
  3. Sentiments and Analogs: FAISS analog lookups and FinBERT sentiment aggregations only ingest embeddings timestamped at or prior to $T$.

### 2.2 Dataset Splitting & Out-of-Sample Partitioning
- **Status:** **PASS WITH CAVEATS**
- **Data Source:** `data/training_dataset.csv` (30,025 rows across 25 tickers).
- **Temporal Window:** 2021-08-05 to 2026-09-08.
- **Split Mechanism:** Chronological split at 80% ($N_{\text{train}} = 24,020$, $N_{\text{test}} = 6,005$).
  - Training window: 2021-08-05 to 2025-09-22.
  - Out-of-sample test window: 2025-09-23 to 2026-09-08.
- **Independence:** Consecutive daily observations per ticker share a 5-day forward target horizon (`future_return_5d`), creating an autocorrelation overlap of 4 days between adjacent daily bars. Observations are temporally ordered, but adjacent rows cannot be treated as statistically independent events.

---

## 3. Confirmed Defects & Leakage Mechanisms

| Defect ID | Severity | File & Location | Description | Impact on Performance |
| :--- | :---: | :--- | :--- | :--- |
| **DEF-01** | **CRITICAL** | `evaluation/paper_trader.py:172` | **Target Peeking in Simulated Trades**: `cand.target == 1` used to filter order entries. | Massive artificial return inflation (+118.44% reported return). |
| **DEF-02** | **HIGH** | `evaluation/calibration.py:130` | **In-Sample Train/Calibration Overlap**: Calibration window $[0.60, 0.80]$ falls inside training set $[0.0, 0.80]$. | Distorted calibration parameters, artificial ECE metrics. |
| **DEF-03** | **HIGH** | `evaluation/paper_trader.py:88` | **Benchmark Substitution**: AAPL substituted for SPY as market benchmark. | Misleading alpha comparison (+98.19 pp reported alpha). |
| **DEF-04** | **MEDIUM** | `evaluation/outcome_tracker.py` | **Zero Persisted Resolution Records**: `prediction_outcomes` and `opportunity_outcomes` tables have 0 rows. | Unpopulated resolution tables; verification relies on raw feature table. |

---

## 4. Deep-Dive: DEF-01 (Target Leakage in Paper Trading)

### Code Inspection
In `evaluation/paper_trader.py` (lines 169–174):
```python
# Select candidates for the day: high predicted probability or committee buy
day_candidates = [
    cand for cand in test_features 
    if cand.date == sim_date and cand.target == 1  # <--- CRITICAL TARGET LEAKAGE!
]
```

### Forensic Analysis
1. In `database/models.py`, `TrainingFeature.target` is defined as:
   $$\text{target} = \begin{cases} 1 & \text{if } \text{future\_return\_5d} > 0 \\ 0 & \text{otherwise} \end{cases}$$
2. At decision timestamp $T$ (market open or close of `sim_date`), `future_return_5d` represents:
   $$\text{future\_return\_5d} = \frac{P_{T+5\text{D}} - P_T}{P_T}$$
3. By filtering on `cand.target == 1`, the paper trading simulator **only buys equities whose forward 5-day return is guaranteed to be strictly positive**!
4. Any candidate whose forward return was negative was filtered out before order execution.
5. **Consequence:** This explains why the paper trading ledger reported an extraordinary +118.44% cumulative return over an 8-month period.

### Reproducible Demonstration
Running the simulator without `cand.target == 1` using only model-predicted probabilities ($\hat{p} \ge 0.60$) or committee recommendations reveals that the realistic strategy return is a fraction of the reported +118.44%.

---

## 5. Deep-Dive: DEF-02 (In-Sample Calibration Contamination)

### Code Inspection
In `evaluation/calibration.py` (lines 126–133):
```python
# Partition into calibration set (20% of history prior to test) and test set
cal_features = features[int(n_samples * 0.60):int(n_samples * 0.80)]
test_features = features[int(n_samples * 0.80):]
```

### Forensic Analysis
1. The champion model artifact `models/artifacts/best_model.pkl` was fitted during Sprint 3 across the entire training slice: $[0.0, 0.80]$ ($24,020$ samples).
2. Consequently, the calibration slice $[0.60, 0.80]$ ($4,804$ samples) was already seen by `best_model.pkl` during training.
3. Fitting Platt Scaling (logistic sigmoid) and Isotonic Regression on in-sample predictions causes severe overfitting to training residuals.
4. Because the raw model already minimized in-sample log-loss during training, its uncalibrated ECE on this contaminated split appeared artificially competitive ($0.0355$).

---

## 6. Deep-Dive: DEF-03 (Benchmark Proxy Substitution)

### Code Inspection
In `evaluation/paper_trader.py` (lines 85–92):
```python
# Obtain benchmark prices (SPY proxy from MarketData)
benchmark_data = session.scalars(
    select(MarketData).where(MarketData.ticker == "AAPL").order_by(MarketData.date.asc())
).all()
```

### Forensic Analysis
1. SPY price bars were not ingested into the local `MarketData` table during Sprint 1.
2. The author substituted `AAPL` as the "SPY proxy" without disclosing the substitution in the final performance report.
3. Over the period 2025-10-01 to 2026-06-01, AAPL returned $+20.25\%$.
4. The reported alpha ($+98.19\%$) was calculated as:
   $$\alpha = 118.44\% - 20.25\% = +98.19\text{ percentage points}$$
5. This is an alpha against Apple Inc. equity, not against the S&P 500 Index (SPY).

---

## 7. Remediation Strategy

1. **Fix DEF-01 in `evaluation/paper_trader.py`:**
   - Remove `cand.target == 1` from candidate selection.
   - Filter trade candidates strictly using decision-time signals:
     $$\hat{p}_{\text{model}} \ge \tau \quad \lor \quad \text{Committee Stance} == \text{"BUY"}$$
2. **Fix DEF-02 in `evaluation/calibration.py`:**
   - Re-partition the training split chronologically:
     - Model Training: $[0.0, 0.60]$ (indices $0$ to $18,015$).
     - Probability Calibration: $[0.60, 0.80]$ (indices $18,015$ to $24,020$).
     - Out-of-Sample Evaluation: $[0.80, 1.0]$ (indices $24,020$ to $30,025$).
   - Fit calibration models only on unseen calibration fold predictions.
3. **Fix DEF-03 in `evaluation/paper_trader.py`:**
   - Compute true benchmark returns using `TrainingFeature.spy_return_1d` (persisted daily SPY return in the feature store) or universe equal-weighted daily return.
4. **Fix DEF-04 in `evaluation/outcome_tracker.py`:**
   - Run batch resolution to populate `prediction_outcomes` and `opportunity_outcomes` with resolved actual returns.

---

## 8. Conclusion

The pipeline features themselves are constructed with sound temporal discipline, but the reported Sprint 9 headline backtest return (+118.44%) was mathematically invalid due to an egregious target leakage filter (`cand.target == 1`). Full restoration of walk-forward causality is required before any strategy claims can be verified.
