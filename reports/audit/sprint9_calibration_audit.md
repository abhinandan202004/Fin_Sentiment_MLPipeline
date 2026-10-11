# Sprint 9.1 — Probability Calibration Audit Report

**Audit Lead:** Senior Quantitative Researcher & Financial Systems Auditor  
**Repository:** `Fin_Sentiment_MLPipeline`  
**Module Audited:** `evaluation/calibration.py`  
**Dataset:** `data/training_dataset.csv` ($N=30,025$, temporal partitions: Train=18,015, Cal=6,005, Eval=6,005)  
**Classification:** Institutional Quantitative Audit  

---

## 1. Executive Summary

Sprint 9 reported that **"Raw uncalibrated probabilities were crowned champion with an Expected Calibration Error (ECE) of 0.0355"**, while Platt Scaling and Isotonic Regression failed to beat the raw model.

### Forensic Finding: **IN-SAMPLE CALIBRATION CONTAMINATION & EVALUATION THRESHOLD LEAKAGE**

1. **Training/Calibration Overlap:**  
   The champion model artifact `models/artifacts/best_model.pkl` was trained on the first $80\%$ of the dataset ($N=24,020$). In `evaluation/calibration.py:158`, the calibration split was defined as indices $[18,015, 24,020]$ (the 60% to 80% partition). Because the model had already observed this calibration slice during training, `prob_cal` reflected in-sample predictions.
2. **Degradation of Post-Processing Calibrators:**  
   Fitting Platt Scaling and Isotonic Regression on in-sample residuals caused severe parameter distortion. When applied to the genuine out-of-sample evaluation set ($N=6,005$):
   - **Raw Baseline:** $\text{ECE} = 0.0355$, $\text{Brier} = 0.2510$
   - **Platt Scaling:** $\text{ECE} = 0.0677$ ($+90.7\%$ error increase), $\text{Brier} = 0.2592$
   - **Isotonic Regression:** $\text{ECE} = 0.1058$ ($+198.0\%$ error increase), $\text{Brier} = 0.2726$
3. **Threshold Leakage Onto Evaluation Set:**  
   In `evaluation/calibration.py:191-192`, `optimize_threshold(y_eval, p_platt)` optimized decision thresholds directly on the test labels `y_eval`.

---

## 2. Quantitative Calibration Comparison Table

| Method | Fitting Data Window | Evaluation Window | ECE ($\downarrow$) | Brier Score ($\downarrow$) | ROC-AUC | Optimal Threshold |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Raw Baseline** | In-Sample ($0-80\%$) | Out-of-Sample ($80-100\%$) | **0.0355** | **0.2510** | **0.5062** | 0.550 (fixed) |
| **Platt Scaling** | Contaminated ($60-80\%$) | Out-of-Sample ($80-100\%$) | **0.0677** | **0.2592** | 0.5062 | 0.400 (peeking) |
| **Isotonic Regression** | Contaminated ($60-80\%$) | Out-of-Sample ($80-100\%$) | **0.1058** | **0.2726** | 0.5011 | 0.400 (peeking) |

---

## 3. Reliability Binning Analysis

The calibration engine partitions evaluation predictions into $M=10$ equal-width probability bins over $[0.0, 1.0]$:

| Bin Range | Center | Count ($N$) | Empirical Accuracy | Mean Confidence | Absolute Bin Calibration Error |
| :---: | :---: | :---: | :---: | :---: | :---: |
| $[0.0, 0.4)$ | 0.35 | 0 | N/A | N/A | $0.0000$ |
| $[0.4, 0.5)$ | 0.45 | 2,752 | 47.16% | 48.21% | **0.0105** |
| $[0.5, 0.6)$ | 0.55 | 3,245 | 57.26% | 51.48% | **0.0578** |
| $[0.6, 0.7)$ | 0.65 | 8 | 87.50% | 61.22% | **0.2628** |
| $[0.7, 1.0]$ | 0.85 | 0 | N/A | N/A | $0.0000$ |

### Why Raw Baseline ECE is 0.0355
The low ECE ($0.0355$) is an artifact of **severe probability clustering around 0.50**:
- $99.87\%$ of all model predictions fall in the narrow range $[0.45, 0.55]$.
- Weighted across sample frequencies:
  $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \big| \text{acc}(B_m) - \text{conf}(B_m) \big| = \frac{2752}{6005}(0.0105) + \frac{3245}{6005}(0.0578) + \frac{8}{6005}(0.2628) \approx 0.0355$$
- The model is not well-calibrated; it is simply uninformative, predicting near 0.50 for virtually all market bars.

---

## 4. Remediation Requirements

1. **Strict 3-Way Temporal Re-training:**  
   To evaluate post-hoc calibrators legitimately:
   - Partition 1 (Train): First 60% ($0$ to $18,015$). Fit base ML model `best_model.pkl` on Partition 1 only.
   - Partition 2 (Calibrate): Middle 20% ($18,015$ to $24,020$). Generate out-of-fold predictions and fit Platt / Isotonic calibrators.
   - Partition 3 (Test): Final 20% ($24,020$ to $30,025$). Evaluate calibrated probabilities strictly on unseen data.
2. **Eliminate Test Threshold Peeking:**  
   In `evaluation/calibration.py`, optimize the decision threshold on Partition 2 (`y_cal`, `p_cal`), never on `y_eval`.
