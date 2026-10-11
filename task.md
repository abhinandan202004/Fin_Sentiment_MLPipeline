# Sprint 10 — Task Checklist: Production Readiness, Risk Monitoring & Paper-Trading Reliability

## Phase 0: Repository Discovery & Baseline Assessment
- [x] Inspect existing architecture, execution path, config, DB models, API routes, model artifacts, and test suite
- [x] Run baseline verification: `python scripts/audit_sprint9_integrity.py` and `python scripts/validate_sprint9_evaluation.py`
- [x] Reconcile 6,249 prediction outcomes vs 6,005 evaluation observations
- [x] Generate deliverable: `reports/sprint10/baseline_assessment.md`

## Phase 1: End-to-End Strategy Integrity & Model Provenance
- [x] Trace complete strategy execution path: Market Data -> Features -> ML Model -> Opportunities -> Committee -> Risk -> Sizing -> Execution -> Ledger -> Outcomes
- [x] Connect ML model prediction ($\hat{p} \ge \text{threshold}$) and Committee governance/veto into paper trading decision loop (replace standalone RSI/sentiment heuristic)
- [x] Audit model-training provenance (chronological splits: 60% Train, 20% Calibrate, 20% Unseen Eval; verify artifact manifest and checksums)
- [x] Generate deliverables: `reports/sprint10/strategy_integrity.md` and `reports/sprint10/model_provenance.md`

## Phase 2: Market Data Reliability & Observability
- [x] Implement data quality validator: stale price detection, missing session detection, holiday handling, price range validation
- [x] Implement feature readiness checks before inference with bounded retries and graceful fallback
- [x] Add operational metrics / telemetry for ingestion freshness, inference latency, and data anomalies
- [x] Generate deliverable: `reports/sprint10/data_reliability.md`

## Phase 3: Central Risk Engine & Capital Protection
- [x] Harden centralized risk engine with configurable limits (max position size, max sector exposure, max portfolio drawdown, cash reserve)
- [x] Implement global trading kill switch with persistent state, audited activation/deactivation, and rejection of new orders
- [x] Enforce order-level pre-trade risk checks (independent of analyst enthusiasm)
- [x] Generate deliverable: `reports/sprint10/risk_controls.md`

## Phase 4: Paper Execution Gateway & Currency Accounting
- [x] Build broker-neutral Paper Execution Gateway (`submit_order`, `get_order`, `cancel_order`, `get_positions`, `get_account_state`)
- [x] Implement explicit order lifecycle: `CREATED -> VALIDATED -> SUBMITTED -> FILLED / REJECTED / CANCELLED` with idempotency keys
- [x] Standardize currency accounting for US equities (explicit USD base currency or documented USD/INR convention) with clean transaction cost deduction
- [x] Generate deliverable: `reports/sprint10/paper_execution.md`

## Phase 5: Portfolio Ledger & Evaluation Integrity
- [x] Implement strict portfolio ledger invariants (`total_equity == cash + market_value_of_positions`, fees deducted exactly once)
- [x] Fully document and reconcile evaluation cohorts (6,005 test observations vs 6,249 stored prediction outcomes)
- [x] Validate benchmark alignment (genuine SPY data without synthetic substitutions)
- [x] Generate deliverable: `reports/sprint10/evaluation_integrity.md`

## Phase 6: Operational Monitoring & Auditability
- [x] Implement / extend health & readiness endpoints (`/health`, `/readiness`, `/status`)
- [x] Add structured audit logging with correlation IDs linking Market Data -> Signal -> Committee -> Order -> Fill -> Outcome
- [x] Generate deliverable: `reports/sprint10/operational_monitoring.md`

## Phase 7: Automated Testing & Failure Simulation
- [x] Implement comprehensive test suite covering Strategy Integrity, Data Reliability, Risk Controls, Execution Lifecycle, and Integration
- [x] Simulate critical failures (stale data, kill switch, missing features, duplicate orders) and verify safe blocks
- [x] Generate deliverable: `reports/sprint10/test_report.md`

## Phase 8: Release Gate & Final Verification
- [x] Create master Sprint 10 release gate validation script: `scripts/validate_sprint10_readiness.py`
- [x] Run full regression suite (Sprint 9 + Sprint 10 tests)
- [x] Generate deliverables: `reports/sprint10/release_gate.md` and `reports/sprint10/known_limitations.md`
- [x] Deliver final Sprint 10 executive summary and paper trading instructions
