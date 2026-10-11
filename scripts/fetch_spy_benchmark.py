"""Fetch the SPY benchmark (explicit, opt-in network call) and store it with provenance.

Usage: python scripts/fetch_spy_benchmark.py [--start 2021-09-01] [--end 2026-09-30]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.benchmark import BenchmarkError, fetch_spy, save_benchmark  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2021-09-01")
    ap.add_argument("--end", default="2026-09-30")
    a = ap.parse_args()
    try:
        rows = fetch_spy(a.start, a.end)
        path = save_benchmark(rows, source="yfinance (unadjusted Close)")
    except BenchmarkError as exc:
        print(f"BLOCKED: {exc}")
        return 1
    print(f"saved {len(rows)} SPY rows ({rows[0][0]}..{rows[-1][0]}) -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
