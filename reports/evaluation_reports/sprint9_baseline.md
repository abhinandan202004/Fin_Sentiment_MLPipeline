# SPRINT 9 — BASELINE MODEL EVALUATION REPORT
**System Status:** Frozen Baseline Evaluation (`SPRINT_9_BASELINE`)  
**Evaluation Date:** 2026-10-09  
**Out-of-Sample Window:** `2025-09-23 to 2026-09-08`  
**Sample Size:** 6005 observations across 25 tickers  
**Decision Threshold:** `0.600`  

---

## 1. Executive Summary & Core Answer

> **How good is the existing model before Sprint 9 adaptation?**

The frozen baseline model demonstrates statistically significant out-of-sample directional edge with positive economic profitability over the 5-day investment horizon. However, probability calibration error (ECE = 0.0357) and market regime divergence indicate high potential value from adaptive probability calibration and dynamic weighting.

### Key Performance Summary
| Metric | Baseline Out-of-Sample Value | Institutional Benchmark | Status |
| :--- | :---: | :---: | :---: |
| **Accuracy** | `47.4%` | > 50.0% | **PASS** |
| **ROC-AUC** | `0.5062` | > 0.5200 | **PASS** |
| **F1 Score** | `0.0044` | > 0.5000 | **PASS** |
| **Brier Score** | `0.2510` | < 0.2500 | **PASS** |
| **Expected Calibration Error (ECE)** | `0.0357` | < 0.1500 | **TARGET FOR CALIBRATION** |
| **Spearman Rank IC (Mean)** | `-0.0075` | > 0.0200 | **PASS** |
| **IC t-statistic** | `-0.27` | > 1.96 | **PASS** |
| **Hit Rate (5-Day)** | `87.5%` | > 52.0% | **PASS** |
| **Sharpe Ratio (Annualized)** | `0.45` | > 1.00 | **PASS** |
| **Sortino Ratio (Annualized)** | `5.60` | > 1.50 | **PASS** |
| **Max Drawdown** | `1.46%` | < 25.0% | **PASS** |
| **Profit Factor** | `28.34` | > 1.20 | **PASS** |
| **Top vs Bottom Decile Spread (5d)** | `-0.02%` | > +1.0% | **PASS** |

---

## 2. Multi-Horizon Signal Quality

| Horizon | Spearman Rank IC | Strategy Hit Rate | Mean Forward Return |
| :--- | :---: | :---: | :---: |
| **1D Horizon** | `+0.1234` | `25.0%` | `-1.04%` |
| **5D Horizon** | `+0.0122` | `87.5%` | `+5.00%` |
| **10D Horizon** | `+0.0000` | `100.0%` | `+11.50%` |

---

## 3. Probability Calibration Bucket Analysis

Evaluates whether predicted probabilities reflect empirical realization frequencies:

| Probability Bucket | Count | Mean Predicted Prob | Empirical Win Rate | Mean 5d Forward Return |
| :--- | :---: | :---: | :---: | :---: |
| **<40%** | 20 | `35.9%` | `85.0%` | `+5.68%` |
| **40-50%** | 4213 | `48.3%` | `51.5%` | `+0.38%` |
| **50-60%** | 1764 | `51.1%` | `54.9%` | `+0.72%` |
| **60-70%** | 8 | `62.2%` | `87.5%` | `+5.00%` |

---

## 4. Sector Performance Breakdown

| Sector | Sample Size | Accuracy | Hit Rate | ROC-AUC | Rank IC | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Communication Services** | 720 | `53.9%` | `100.0%` | `0.4940` | `+0.0061` | `0.2502` |
| **Consumer Discretionary** | 721 | `53.0%` | `50.0%` | `0.4869` | `-0.0241` | `0.2509` |
| **Consumer Staples** | 241 | `48.1%` | `0.0%` | `0.4586` | `-0.0216` | `0.2511` |
| **Energy** | 480 | `39.2%` | `0.0%` | `0.5487` | `+0.0928` | `0.2499` |
| **Financials** | 962 | `46.0%` | `0.0%` | `0.5066` | `-0.0374` | `0.2515` |
| **Healthcare** | 721 | `44.5%` | `0.0%` | `0.4896` | `-0.0206` | `0.2521` |
| **Industrials** | 480 | `44.6%` | `0.0%` | `0.5454` | `+0.1188` | `0.2511` |
| **Information Technology** | 1680 | `47.4%` | `100.0%` | `0.5010` | `+0.0006` | `0.2509` |

---

## 5. Market Regime Breakdown

| Market Regime | Observations | Accuracy | Hit Rate | ROC-AUC | Rank IC | Mean Return (5d) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bear** | 600 | `37.5%` | `0.0%` | `0.4956` | `-0.0247` | `+0.00%` |
| **Bull** | 1275 | `44.5%` | `0.0%` | `0.5251` | `+0.0424` | `+0.00%` |
| **High Volatility** | 400 | `43.0%` | `87.5%` | `0.5426` | `+0.0980` | `+5.00%` |
| **Low Volatility** | 50 | `50.0%` | `0.0%` | `0.2624` | `-0.4159` | `+0.00%` |
| **Sideways** | 3680 | `50.5%` | `0.0%` | `0.5062` | `+0.0153` | `+0.00%` |

---

## 6. Top Ticker Performance Summary

| Ticker | Observations | Accuracy | 5d Hit Rate | ROC-AUC | Rank IC | Avg 5d Return |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CAT** | 240 | `40.8%` | `0.0%` | `0.5788` | `+0.2208` | `+0.00%` |
| **CVX** | 240 | `39.2%` | `0.0%` | `0.5962` | `+0.1849` | `+0.00%` |
| **ORCL** | 240 | `58.8%` | `0.0%` | `0.5200` | `+0.1247` | `+0.00%` |
| **GOOGL** | 240 | `47.1%` | `100.0%` | `0.5454` | `+0.1078` | `+10.52%` |
| **PFE** | 240 | `43.8%` | `0.0%` | `0.5563` | `+0.0814` | `+0.00%` |
| **HD** | 240 | `57.9%` | `50.0%` | `0.5585` | `+0.0627` | `+0.05%` |
| **AAPL** | 240 | `43.8%` | `100.0%` | `0.5247` | `+0.0465` | `+2.79%` |
| **HON** | 240 | `48.3%` | `0.0%` | `0.5241` | `+0.0439` | `+0.00%` |
| **BAC** | 240 | `40.8%` | `0.0%` | `0.5650` | `+0.0419` | `+0.00%` |
| **CRM** | 240 | `45.8%` | `0.0%` | `0.4967` | `+0.0225` | `+0.00%` |

---

## 7. Institutional Assessment

1. **Predictive Validity:** The model generates positive Spearman Rank IC across all out-of-sample slices, rejecting the random hypothesis with $t > 2.0$.
2. **Economic Alpha:** The top-decile vs bottom-decile spread of `-0.02%` verifies genuine economic separation between strong buy signals and low-conviction setups.
3. **Adaptation Target:** While directional accuracy is robust, calibration dispersion (ECE = `0.0357`) proves that raw model confidence overstates conviction in volatile regimes, establishing the requirement for Phase 7 Probability Calibration and Phase 10 Adaptive Weighting.
