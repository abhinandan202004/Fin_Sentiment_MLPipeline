# Committee Performance Attribution Report
**Generated:** 2026-09-20 21:23:30 UTC  
**System:** Fin_Sentiment_MLPipeline — Sprint 7: Committee Learning & Performance Attribution

---

## 1. Outcome Resolution Summary

| Metric | Value |
|--------|-------|
| **Total Decisions** | 65 |
| **Resolved** | 55 |
| **Pending** | 10 |
| **Resolution Rate** | 85% |

### Recent Resolutions
- ⏳ **NVDA**: Only 6d old, need 7d
- ⏳ **NVDA**: Only 6d old, need 7d
- ⏳ **TCS**: Only 6d old, need 7d
- ⏳ **NVDA**: Only 4d old, need 7d
- ⏳ **JPM**: Only 4d old, need 7d
- ⏳ **JPM**: Only 4d old, need 7d
- ⏳ **NVDA**: Only 4d old, need 7d
- ⏳ **AAPL**: Only 4d old, need 7d
- ⏳ **XOM**: Only 4d old, need 7d
- ⏳ **JPM**: Only 4d old, need 7d

---

## 2. Agent Attribution — Performance Leaderboard

| Agent | Accuracy | Brier Score | Calibration Error | Avg DD on Miss | Samples | Trend | Status |
|-------|:--------:|:-----------:|:-----------------:|:--------------:|:-------:|:-----:|:------:|
| **EvidenceProsecutor** | 51% | 0.367 | 0.338 | -5.10% | 55 | 📉 degrading | live_tracked |
| **RiskOfficer** | 52% | 0.303 | 0.232 | -4.34% | 54 | ➡️ stable | live_tracked |
| **BullAnalyst** | 100% | 0.222 | 0.458 | — | 54 | ➡️ stable | live_tracked |
| **BearAnalyst** | 50% | 0.168 | 0.042 | -1.92% | 54 | 📈 improving | live_tracked |
| **PortfolioManager** | 55% | — | — | — | 0 | — insufficient_data | baseline |

> **Best Agent:** `BullAnalyst` — 100% accuracy (54 samples)

---

## 3. Evidence Stream Reliability

| Signal Stream | Accuracy | Hit Rate | Correct | Incorrect | Avg DD on Miss | Samples |
|---------------|:--------:|:--------:|:-------:|:---------:|:--------------:|:-------:|
| **ML_MODEL** | 100% | 100% | 55 | 0 | — | 55 |
| **ANALOG_ENGINE** | 98% | 98% | 54 | 1 | — | 55 |
| **TECHNICAL** | 76% | 76% | 42 | 13 | -1.87% | 55 |
| **SENTIMENT** | 49% | 49% | 27 | 28 | -5.10% | 55 |

> **Most Reliable Signal:** `ML_MODEL` — 100% accuracy

---

## 4. Regime Performance Analysis

| Regime | Decisions | Accuracy | Avg 5d | Avg 10d | Avg 20d | Avg DD | Max DD | Post-Dec Vol |
|--------|:---------:|:--------:|:------:|:-------:|:-------:|:------:|:------:|:------------:|
| **Unknown** | 1 | 100% | -1.20% | — | -2.50% | — | — | — |
| **Bull** | 16 | 75% | +1.92% | +2.37% | +2.51% | -3.02% | -6.39% | 21.85% |
| **Bear** | 8 | 88% | -0.15% | +0.54% | +0.66% | -3.66% | -6.34% | 22.97% |
| **High Volatility** | 7 | 71% | +0.33% | +1.00% | +1.75% | -3.44% | -6.03% | 22.81% |
| **Low Volatility** | 10 | 60% | +1.72% | +1.74% | +2.92% | -3.31% | -6.33% | 21.06% |
| **Sideways** | 13 | 92% | -1.06% | -0.91% | -0.71% | -4.22% | -6.99% | 20.29% |

### Agent × Regime Accuracy Matrix

| Regime | **BearAnalyst** | **BullAnalyst** | **EvidenceProsecutor** | **RiskOfficer** |
|--------|:------:|:------:|:------:|:------:|
| **Unknown** | — | — | 100% (n=1) | — |
| **Bull** | 31% (n=16) | 100% (n=16) | 69% (n=16) | 69% (n=16) |
| **Bear** | 62% (n=8) | 100% (n=8) | 38% (n=8) | 62% (n=8) |
| **High Volatility** | 43% (n=7) | 100% (n=7) | 57% (n=7) | 43% (n=7) |
| **Low Volatility** | 40% (n=10) | 100% (n=10) | 60% (n=10) | 60% (n=10) |
| **Sideways** | 77% (n=13) | 100% (n=13) | 23% (n=13) | 23% (n=13) |

---

## 5. Adaptive Voting Weights

**Status:** `ADAPTIVE`  
**Resolved Decisions:** 55

| Agent | Current Weight | EWMA Accuracy | Baseline Weight |
|-------|:--------------:|:-------------:|:---------------:|
| **BullAnalyst** | 41.33% ↑ | 100.00% | 35.00% |
| **BearAnalyst** | 24.97% → | 60.42% | 25.00% |
| **RiskOfficer** | 17.30% ↓ | 41.85% | 25.00% |
| **EvidenceProsecutor** | 16.41% ↑ | 39.70% | 15.00% |

### Weight Evolution (Last 5 Checkpoints)

| Checkpoint | BullAnalyst | BearAnalyst | RiskOfficer | EvidenceProsecutor |
|:----------:|:-----------:|:-----------:|:-----------:|:------------------:|
| @35 decisions | 38.88% | 18.29% | 21.72% | 21.10% |
| @40 decisions | 38.45% | 16.76% | 22.68% | 22.11% |
| @45 decisions | 39.13% | 17.79% | 21.34% | 21.74% |
| @50 decisions | 39.36% | 19.28% | 20.89% | 20.47% |
| @55 decisions | 39.56% | 19.78% | 20.51% | 20.14% |

---

## 6. Counterfactual Analysis — Is the Committee Worth It?

### Strategy Comparison

| Strategy | Accuracy | Avg 5d Return | Avg Drawdown | Sharpe Proxy | Decisions |
|----------|:--------:|:-------------:|:------------:|:------------:|:---------:|
| **Committee (Actual)** | 78% | +0.62% | -3.51% | 0.18 | 55 |
| ML Model Only | 100% | +0.62% | -3.51% | 0.18 | 55 |
| Consensus Signals | 98% | +0.62% | -3.51% | 0.18 | 55 |

### Committee vs Individual Agents

| Agent | Accuracy | Avg 5d Return | Avg Drawdown | Sharpe Proxy |
|-------|:--------:|:-------------:|:------------:|:------------:|
| EvidenceProsecutor | 51% | +0.62% | -3.51% | 0.18 |
| BullAnalyst | 100% | +0.65% | -3.51% | 0.19 |
| BearAnalyst | 50% | +0.65% | -3.51% | 0.19 |
| RiskOfficer | 52% | +0.65% | -3.51% | 0.19 |

### Edge Analysis

| Comparison | Accuracy Edge | Return Edge | Drawdown Edge | Verdict |
|------------|:------------:|:-----------:|:-------------:|:-------:|
| committee_vs_ml | -21.82% | +0.00% | +0.00% | `BENCHMARK_OUTPERFORMS` |
| committee_vs_consensus | -20.00% | +0.00% | +0.00% | `BENCHMARK_OUTPERFORMS` |
| committee_vs_best_agent | -21.82% | -0.03% | — | `AGENT_OUTPERFORMS` |

### 🏛️ Overall Verdict

> **COMMITTEE_QUESTIONABLE — Simpler strategies outperform; consider simplification**

---

## 7. System Status

| Component | Status |
|-----------|--------|
| **Outcome Resolution** | 55/65 resolved |
| **Agent Attribution** | 217 evaluated votes |
| **Evidence Attribution** | 55 decisions analyzed |
| **Adaptive Weights** | `ADAPTIVE` (55 decisions) |
| **Counterfactual Analysis** | `COMPUTED` |

---

*Report generated by Sprint 7 — Committee Learning & Performance Attribution.*  
*Fin_Sentiment_MLPipeline — 2026-09-20 21:23:30 UTC*
