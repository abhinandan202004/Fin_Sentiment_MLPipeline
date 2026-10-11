# Sprint 9.1 — ML Prediction Metrics Audit Report

**Audit Lead:** Senior Quantitative Researcher & Financial Systems Auditor  
**Repository:** `Fin_Sentiment_MLPipeline`  
**Dataset:** `data/training_dataset.csv` (Out-of-Sample Split: $N=6,005$ from 2025-09-23 to 2026-09-08)  
**Model Under Test:** `models/artifacts/best_model.pkl` (Frozen Baseline)  
**Classification:** Institutional Quantitative Audit  

---

## 1. Executive Summary

Sprint 9 reported an extraordinary **"87.5% prediction hit rate at threshold"** across an out-of-sample test set of 6,005 observations.

### Forensic Finding: **THE 87.5% HIT RATE IS A MICROSCOPIC SUBSET ARTIFACT ($N=8$)**

The reported 87.5% hit rate represents exactly **7 correct predictions out of only 8 observations** that crossed the confidence threshold ($\hat{p} \ge 0.60$). 
- Selection frequency: Only **0.133%** (8 out of 6,005) of test observations were selected.
- True Model Recall: **0.22%** ($7 / 3,162$ positive return events).
- Overall Out-of-Sample Accuracy: **47.44%**, which **underperforms the naive majority-class baseline of 52.66%**.
- ROC-AUC: **0.5062** (statistically indistinguishable from random chance $0.50$).
- Rank Information Coefficient (IC): **$-0.0075$** with a t-statistic of **$-0.271$** ($p > 0.70$).
- Decile Spread: Top decile (+1.15%) underperformed the bottom decile (+1.17%) by $-2.2$ bps over 5 days.

Presenting an 87.5% hit rate alongside a sample size of 6,005 observations created the false impression that the model achieved 87.5% accuracy across 6,005 decisions. In institutional reporting, this is classified as a **Severe Denominator Concealment Defect**.

---

## 2. Comprehensive Out-of-Sample Statistical Metrics

All metrics independently recomputed on the frozen 20% out-of-sample partition ($N=6,005$, 25 equities):

| Metric | Headline / Stored Value | Independently Recomputed | Institutional Benchmark / Notes |
| :--- | :---: | :---: | :--- |
| **Out-of-Sample $N$** | 6,005 | **6,005** | Confirmed (2025-09-23 to 2026-09-08). |
| **Prediction Threshold** | 0.60 | **0.60** | Defined in `optimal_threshold.json`. |
| **Observations $\ge 0.60$** | *Unreported* | **8 (0.133%)** | **True denominator of 87.5% hit rate**. |
| **Hit Rate at Threshold** | 87.5% | **87.50% (7 / 8)** | Restricted to $N=8$. |
| **Overall Accuracy ($\tau=0.50$)** | 47.4% | **47.44%** | Fails naive majority baseline. |
| **Balanced Accuracy** | *Unreported* | **47.41%** | Symmetric true positive/negative rate. |
| **Majority-Class Baseline** | *Unreported* | **52.66%** | $3,162 / 6,005$ positive return events. |
| **Precision ($\tau=0.60$)** | 87.5% | **87.50%** | Equal to hit rate for binary classification. |
| **Recall ($\tau=0.60$)** | *Unreported* | **0.22%** | Captures only 7 of 3,162 opportunities. |
| **F1 Score ($\tau=0.60$)** | *Unreported* | **0.0044** | Near zero due to microscopic recall. |
| **ROC-AUC** | *Unreported* | **0.5062** | Barely above coin flip (0.5000). |
| **Brier Score** | 0.2510 | **0.2510** | Equivalent to uninformative prior ($0.5^2 = 0.25$). |
| **Log Loss** | 0.6952 | **0.6952** | Near random entropy ($-\ln 0.5 \approx 0.6931$). |
| **Expected Calibration Error**| 0.0357 | **0.0357** | Low ECE caused by probability compression near 0.50. |
| **Rank IC (Pooled)** | *Unreported* | **+0.0122** | Weak linear rank correlation. |
| **Mean Monthly Rank IC** | *Unreported* | **$-0.0075$** | Negative mean across rolling windows. |
| **IC Standard Deviation** | *Unreported* | **0.0999** | High temporal variance. |
| **IC t-Statistic** | *Unreported* | **$-0.2711$** | Not statistically significant ($p = 0.789$). |
| **Information Ratio (IR)** | *Unreported* | **$-1.1934$** | Negative risk-adjusted information decay. |

---

## 3. Confusion Matrix Breakdown

### Default Decision Threshold ($\tau = 0.50$)
$$\begin{pmatrix} \text{TN} & \text{FP} \\ \text{FN} & \text{TP} \end{pmatrix} = \begin{pmatrix} 2835 & 8 \\ 3148 & 14 \end{pmatrix}$$
- Total Predictions: 6,005
- Negative Class ($y=0$): 2,843 (47.34%)
- Positive Class ($y=1$): 3,162 (52.66%)
- **Model Bias:** The model exhibits severe prediction compression into the $[0.48, 0.52]$ interval, predicting $\hat{p} < 0.50$ for 5,983 of 6,005 samples.

### Calibrated Threshold ($\tau = 0.60$)
$$\begin{pmatrix} \text{TN} & \text{FP} \\ \text{FN} & \text{TP} \end{pmatrix} = \begin{pmatrix} 2842 & 1 \\ 3155 & 7 \end{pmatrix}$$
- True Positives ($TP$): 7
- False Positives ($FP$): 1
- False Negatives ($FN$): 3,155
- True Negatives ($TN$): 2,842
- $\text{Hit Rate} = \frac{TP}{TP + FP} = \frac{7}{7 + 1} = 87.5\%$
- $\text{Recall} = \frac{TP}{TP + FN} = \frac{7}{3162} = 0.00221 \ (0.22\%)$

---

## 4. Decile Return Monotonicity Analysis

A robust quantitative alpha factor must demonstrate monotonic return ordering across predicted decile buckets:

| Decile Bucket | Predicted Prob Range | 5-Day Realized Return | Spread vs Bottom Decile |
| :--- | :---: | :---: | :---: |
| **Decile 1 (Bottom)** | $[0.000, 0.450)$ | **+1.172%** | Baseline |
| **Decile 2-9 (Middle)** | $[0.450, 0.520)$ | **+1.085%** | $-8.7$ bps |
| **Decile 10 (Top)** | $[0.520, 1.000]$ | **+1.150%** | **$-2.2$ bps** |

- **Decile Spread ($Q_{10} - Q_1$):** $-0.022\%$ ($-2.2$ basis points).
- **Verdict:** Monotonicity fails completely. Equities sorted into the lowest probability decile generated higher 5-day returns than those sorted into the highest probability decile.

---

## 5. Performance by Market Regime and Sector

### Market Regime Breakdown
| Regime | Observations | Accuracy ($\tau=0.50$) | Signals $\ge 0.60$ | Hit Rate ($\tau=0.60$) | Rank IC | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bull** | 1,275 | 44.55% | 0 | N/A (0.0%) | +0.0424 | 0.5251 |
| **Bear** | 600 | 37.50% | 0 | N/A (0.0%) | $-0.0247$ | 0.4956 |
| **Sideways** | 3,680 | 50.52% | 0 | N/A (0.0%) | +0.0153 | 0.5062 |
| **High Volatility** | 400 | 43.00% | **8** | **87.50%** | +0.0980 | 0.5426 |
| **Low Volatility** | 50 | 50.00% | 0 | N/A (0.0%) | $-0.4159$ | 0.2624 |

**Finding:** All 8 trades crossing the 0.60 threshold occurred exclusively during the 400 observations of the "High Volatility" regime. During all normal market regimes (Bull, Bear, Sideways), the model did not generate a single trade signal.

### Sector Breakdown
| Sector | Observations | Accuracy | Signals $\ge 0.60$ | Sector Hit Rate | Sector Brier | Sector Rank IC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Information Tech** | 1,680 | 47.44% | 4 | 100.0% (4/4) | 0.2509 | +0.0006 |
| **Communication Serv**| 720 | 53.89% | 2 | 100.0% (2/2) | 0.2502 | +0.0061 |
| **Consumer Discretionary**| 721 | 52.98% | 2 | 50.0% (1/2) | 0.2509 | $-0.0241$ |
| **Financials** | 962 | 46.05% | 0 | N/A | 0.2515 | $-0.0374$ |
| **Healthcare** | 721 | 44.52% | 0 | N/A | 0.2521 | $-0.0206$ |
| **Industrials** | 480 | 44.58% | 0 | N/A | 0.2511 | +0.1188 |
| **Energy** | 480 | 39.17% | 0 | N/A | 0.2499 | +0.0928 |
| **Consumer Staples** | 241 | 48.13% | 0 | N/A | 0.2511 | $-0.0216$ |

---

## 6. Audit Conclusion & Recommendations

1. **Retract the Claim of "87.5% Hit Rate Across 6,005 Observations":**  
   The true hit rate at threshold is $87.5\%$ on a sample size of $N=8$. Transparently report the sample size, selection rate (0.133%), and recall (0.22%).
2. **Acknowledge Sub-Baseline Model Discrimination:**  
   With ROC-AUC $= 0.5062$ and negative rank IC ($-0.0075$), the raw ML model possesses no statistical edge over the universe majority baseline.
3. **Threshold Recalibration Required:**  
   A threshold of 0.60 is excessively conservative for a model with compressed probability outputs. The system must use ranked cross-sectional percentiles or calibrate probabilities to restore operational signal generation across market regimes.
