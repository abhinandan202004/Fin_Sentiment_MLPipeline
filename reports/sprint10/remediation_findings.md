# Sprint 10.1 — Remediation Findings (code-verified)

Status values: CONFIRMED = reproduced by reading code; each has a fix + test below.

| # | Defect | Evidence | Fix |
|---|--------|----------|-----|
| 1 | Orders/idempotency in memory only | `paper_gateway.py` dicts | `execution/order_store.py` (SQLite, UNIQUE idempotency key) + fill replay |
| 2 | Slippage charged twice | fill price shifted 2 bps AND ledger deducted 2 bps slippage fee | slippage embedded in fill price only; fees = commission + spread |
| 3 | No locking | no lock around risk→fill | `threading.RLock` in `submit_order` |
| 4 | Non-atomic ledger | `assert` after cash mutation | validate first, mutate after; no post-mutation asserts |
| 5 | Tautological invariant | `verify_ledger_invariant` compared `cash+pos` to itself | `reconcile()` rebuilds cash from the transaction history |
| 6 | BUY of an open ticker overwrote the position | `open_positions[ticker] = pos` | explicit rejection |
| 7 | SELL ignored order quantity | `record_sell` closes whole position | quantity must equal held shares, else rejected |
| 8 | Release gate is a smoke test | stage 2 prints hashes only; "100% READY" is hardcoded; stage 4 deactivates the real kill switch without audit | gate rewrite (step G, pending) |
| 9 | Tests mutate shared kill-switch file | `setUp` calls `deactivate_kill_switch` | pending (step D) |
| 10 | No API routes / SPY benchmark / model manifest | not found | pending (steps E, F); provenance stays UNVERIFIED |

Defects 8–10 remain **open**; the Sprint 10 "PASS" verdict is withdrawn until they are closed.
