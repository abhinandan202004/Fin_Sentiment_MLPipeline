# Sprint 10 — Known Limitations & Operational Risk Disclosures

**Audit Date:** 2026-10-10  
**Scope:** Transparent Disclosure of Modeling, Microstructure, and Environmental Constraints  

---

## 1. Overview

In accordance with institutional risk management standards, this document registers the known architectural, mathematical, and environmental limitations of `Fin_Sentiment_MLPipeline` following Sprint 10 hardening.

---

## 2. Documented Limitations

### 2.1 Microstructure & Fill Assumptions
- **Execution Assumption:** Paper orders assume simulated fills based on contemporaneous daily closing/open prices adjusted by $2\text{ bps}$ slippage and $3\text{ bps}$ spread.
- **Limitation:** In live markets, intraday liquidity fragmentation, market impact on large orders, and market open auction imbalances can generate wider real-world execution decay, particularly during earnings announcements or high-VIX macro regimes.

### 2.2 Universe Coverage Scope
- **Current Universe:** Exactly 25 large-cap US equities (e.g. AAPL, NVDA, MSFT, JPM, XOM).
- **Limitation:** All assets are highly liquid mega-cap equities with active options and cash markets. Findings and backtested returns cannot be generalized to small-cap or illiquid equities without re-calibrating liquidity filters and market impact models.

### 2.3 Database Backend & Environment Fallback
- **Active Backend:** SQLite database at `sqlite:///data/fin_sentiment.db`.
- **Limitation:** PostgreSQL credentials on `localhost:5432` failed authentication (`user: postgres`), triggering the automatic local SQLite fallback. While SQLite satisfies all single-process ACID transactions and paper simulation requirements, multi-threaded high-throughput execution in cloud environments requires PostgreSQL.

### 2.4 Maturing Forward Outcomes in Recent Cohorts
- **Horizon Maturity:** Across the 6,005 test fold predictions, the 1d, 5d, 10d, and 20d horizons are 100% resolved. However, 1,500 predictions in the 60-day horizon remain marked as `PENDING` because fewer than 60 trading bars have elapsed between late July 2026 and the September 8, 2026 dataset boundary.

### 2.5 Static Model Checkpoint
- **Model Training State:** The champion Random Forest (`best_model.pkl`) was fitted on historical training data up to late 2024.
- **Limitation:** The model is not continuously retrained in real time. Model weights remain frozen to maintain walk-forward evaluation validity. Continuous automated retraining pipelines are reserved for future architectural releases.

---

## 3. Operational Safeguards in Place

- **No Live Broker Connectivity:** Live broker order routing is completely absent from the codebase; the platform runs in strictly simulated paper mode.
- **Global Trading Kill Switch:** Any operational anomaly can be arrested immediately via `RiskGuard.activate_kill_switch()`.
- **Idempotency Gate:** Network retries cannot produce duplicate order submissions or double fee deductions.
- **Stale Data Interceptor:** Missing market bars automatically block new order generation.
