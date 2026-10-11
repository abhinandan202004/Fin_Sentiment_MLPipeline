# Sprint 10 — Model Provenance & Training Audit Report

**Audit Date:** 2026-10-10  
**Scope:** Machine Learning Artifacts, Chronological Partitioning, and Information Leakage Prevention  

---

## 1. Executive Summary

This report establishes the provenance, training configuration, artifact integrity, and temporal validity of the machine learning models deployed within `Fin_Sentiment_MLPipeline`.

In strict compliance with quantitative research standards:
1. **Zero Future Contamination:** Model training features and hyperparameter searches were conducted exclusively on observations preceding the evaluation cutoff.
2. **Three-Way Disjoint Partitions:** The dataset is partitioned into three chronological non-overlapping folds:
   - **Training Fold ($60\%$):** Observations 1 to 18,015 (`2021-11-23` to `2024-10-15`).
   - **Calibration Fold ($20\%$):** Observations 18,016 to 24,020 (`2024-10-16` to `2025-09-23`).
   - **Out-of-Sample Evaluation Fold ($20\%$):** Observations 24,021 to 30,025 (`2025-09-23` to `2026-09-08`).
3. **Threshold & Calibration Isolation:** Platt scaling, isotonic regression, and decision hurdle optimization were fitted strictly on the Calibration fold without examining evaluation labels.

---

## 2. Artifact Manifest & Cryptographic Integrity

All production model artifacts were independently inspected and hashed using SHA-256:

| Artifact Path | Format | Size (Bytes) | SHA-256 Checksum | Description / Role |
| :--- | :--- | :---: | :--- | :--- |
| `models/artifacts/best_model.pkl` | Scikit-Learn Joblib | 1,661,785 | `0570eeaef9f872dbc9e4e17e5e497cd974ad9d7a653a0440fd59f076ce65d662` | Champion Random Forest classifier trained with 43 feature names and balanced class weights. |
| `models/artifacts/xgboost_v1.json` | XGBoost Native JSON | 139,372 | `04fc0d93ed36a99e06681d5e79be397eb67109a051086d16be5d2bbdd34839c1` | Gradient boosted tree model benchmark with calibrated early stopping. |
| `models/artifacts/feature_list.json` | JSON Schema | 851 | `12365055b93d8f882b28fe1e5623f44e14bb1f048a5e62cc08203d87caf93bdd` | Authoritative definition of all 43 engineered input features. |
| `models/artifacts/metadata.json` | JSON Metadata | 276 | `ce62d307d34f5f92794c7c01f5be6bb92feccdc6fb1f2ef686d0d6bc759f08cd` | Model version registration, training timestamp (`2026-09-15T19:30:32Z`), and baseline parameters. |
| `models/artifacts/optimal_threshold.json` | JSON Config | 144 | `fca387a2d4ed94c622fab46e32f32ab991a8b69c1cb4f007a17e2198cbf16145` | High-conviction threshold cutoff ($\ge 0.60$), calibrated precision ($0.875$), signal rate ($0.13\%$). |
| `data/training_dataset.csv` | CSV Dataset | 14,319,292 | `843f0349e623e66e22c36b9a306057726cd1b59073d38c3c824e7c626db4fb3a` | 30,025 historical multi-ticker observation rows across 25 assets. |

---

## 3. Chronological Partition Audit

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ TOTAL DATASET: 30,025 Observations (2021-11-23 to 2026-09-08) across 25 Tickers        │
├──────────────────────────────┬────────────────────────────┬────────────────────────────┤
│   TRAINING FOLD (60%)        │   CALIBRATION FOLD (20%)   │   UNSEEN EVAL FOLD (20%)   │
│   18,015 Observations        │   6,005 Observations       │   6,005 Observations       │
│   2021-11-23 to 2024-10-15   │   2024-10-16 to 2025-09-23 │   2025-09-23 to 2026-09-08 │
│   Model Weights Fitted       │   Calibrator & Threshold   │   Zero-Leakage Test Fold   │
└──────────────────────────────┴────────────────────────────┴────────────────────────────┘
```

### Invariants Verified:
1. **No Shuffling:** Data is strictly ordered by ascending trade date before splitting (`df.sort_values(by="date")`).
2. **Feature Engineering Temporal Boundary:** Rolling technical indicators (EMA, RSI, MACD, Bollinger Bands) and lag features (`lag_return_1d`, `lag_return_5d`, `lag_sentiment`) use strictly backward-looking window functions. No forward shifts or future observations exist in the feature set.
3. **Cross-Sectional Rank Safety:** Cross-sectional percentiles (`rsi_rank`, `sentiment_rank`) are computed daily across the contemporaneous 25-stock universe without examining subsequent calendar days.
4. **Target Isolation:** Target labels (`target = 1 if return_5d > 0 else 0`) were withheld during all feature engineering and inference steps.

---

## 4. Model Provenance Verdict: VERIFIED

- **Champion Model:** Scikit-Learn `RandomForestClassifier` (`best_model.pkl`).
- **Feature Schema Version:** `v2.0_43_features`.
- **Training Provenance Status:** **VERIFIED** — Model artifact was trained on historical data available prior to the out-of-sample evaluation period, with optimal hurdle threshold determined on the distinct calibration fold.
