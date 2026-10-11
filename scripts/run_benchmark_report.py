"""Compute strategy-vs-SPY metrics from persisted Sprint 9.2 paper-trading snapshots.

IMPORTANT: these snapshots come from the legacy simulation (INR-denominated cash, built
before the Sprint 10 gateway). Returns are unitless so currency does not affect them,
but the run itself is NOT a Sprint 10 gateway run and its trades predate the leakage fix
verification. Results are labelled accordingly in the output.

Usage: python scripts/run_benchmark_report.py
"""
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text  # noqa: E402

from database.connection import SessionLocal  # noqa: E402
from evaluation.benchmark import BenchmarkError, compute_performance, load_benchmark  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "reports" / "sprint10" / "benchmark_metrics.json"


def main() -> int:
    s = SessionLocal()
    try:
        rows = s.execute(text(
            "SELECT snapshot_date, total_equity FROM paper_trading_snapshots ORDER BY snapshot_date"
        )).fetchall()
    finally:
        s.close()
    equity = {}
    for d, v in rows:
        equity[date.fromisoformat(str(d)[:10])] = float(v)  # last snapshot per day wins
    equity_series = sorted(equity.items())

    try:
        bench, meta = load_benchmark()
    except BenchmarkError as exc:
        bench, meta = None, {"error": str(exc)}

    result = compute_performance(equity_series, bench)
    result["equity_source"] = ("legacy Sprint 9.2 paper_trading_snapshots "
                               "(not a Sprint 10 gateway run; selection logic provenance: see strategy_integrity.md)")
    result["benchmark_provenance"] = meta
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))
    return 0 if result["status"] == "OK" else 2


if __name__ == "__main__":
    sys.exit(main())
