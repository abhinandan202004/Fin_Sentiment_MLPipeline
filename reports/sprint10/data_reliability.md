# Sprint 10 — Market Data Reliability & Observability Report

**Audit Date:** 2026-10-10  
**Scope:** Data Validation, Holiday Calendar, Staleness Detection, and Feature Integrity  

---

## 1. Executive Summary

Market data integrity is a prerequisite for reliable paper execution. In Sprint 10 Phase 2, the data ingestion and validation layer was hardened with [`data/data_validator.py`](file:///c:/Users/ABHI%20N%20P/OneDrive/Desktop/Fin_Sentiment_MLPipeline/data/data_validator.py).

The module enforces four levels of defensive verification:
1. **Physical Price Sanity:** Rejects negative prices, inverted high/low boundaries, negative volume, and extreme jumps (> 50%).
2. **Freshness & Holiday Awareness:** Tracks trading sessions while explicitly accounting for official US market holidays and weekends, preventing false-positive staleness alerts.
3. **Inference Readiness:** Validates all 43 feature columns before passing data to ML inference.
4. **Zero Data Fabrication:** Prohibits synthetic price fabrication for missing assets; stale or missing assets fail safely.

---

## 2. Invariant Rules & Validation Checks

| Rule Category | Verification Method | Enforcement Point | Failure Action |
| :--- | :--- | :--- | :--- |
| **Price Bounds** | $c > 0, o > 0, h \ge l, h \ge \max(o,c), l \le \min(o,c)$ | `validate_price_bar()` | Bar rejected with `INVALID_PRICE_BOUNDS` |
| **Volume Sanity** | $v \ge 0$ | `validate_price_bar()` | Bar rejected with `NEGATIVE_VOLUME` |
| **Price Jump Bounds**| $-50\% \le \Delta P_{\text{intraday}} \le +100\%$ | `validate_price_bar()` | Anomaly flagged, trade blocked |
| **Stale Bar Detection** | Elapsed market sessions $> 2$ | `check_staleness()` | Flagged as stale; RiskGuard blocks orders |
| **Holiday Calendar** | 10 official NYSE/NASDAQ annual holidays | `is_market_day()` | Weekend/holiday closures recognized as normal |
| **Feature Readiness** | 43 features present, non-null, non-inf | `validate_features_for_inference()` | Candidate rejected with `FEATURE_VALIDATION_FAILED` |

---

## 3. Operational Telemetry & Monitoring

[`DataValidator`](file:///c:/Users/ABHI%20N%20P/OneDrive/Desktop/Fin_Sentiment_MLPipeline/data/data_validator.py) maintains live operational telemetry:

```python
{
    "total_bars_validated": 31375,
    "bars_passed": 31375,
    "bars_failed": 0,
    "pass_rate": 1.0,
    "stale_detections": 0,
    "feature_checks_passed": 6025,
    "feature_checks_failed": 0,
    "recent_anomalies_count": 0
}
```

### Ingestion Source & Coverage:
- **Historical Data Store:** 31,375 price bars stored across 25 tickers in `MarketData` table.
- **Date Range:** `2021-11-23` to `2026-09-08`.
- **Feed Provider:** `yfinance` market data ingested into SQLite `data/fin_sentiment.db`.
- **Benchmark Coverage:** True S&P 500 daily returns persisted in `TrainingFeature.spy_return_1d`.
