"""
Genuine SPY benchmark + reproducible performance metrics (Sprint 10.1, step E).

Rules enforced here:
  * The benchmark is SPY only. No other ticker is ever substituted. If valid SPY data
    is unavailable, results are reported as status="BLOCKED" with the reason.
  * Strategy and benchmark returns are aligned on the exact intersection of dates;
    dropped dates and coverage are reported, never forward-filled or interpolated.
  * Annualization is only reported when the sample is long enough (min_obs_annualize).
  * Benchmark data is stored OUTSIDE the 25-ticker market_data table (data/benchmark/)
    with a provenance sidecar (source, fetch time, row count, sha256).
Metrics describe a historical run only; they are not forecasts.
"""

import hashlib
import json
import logging
import math
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger("Benchmark")

BENCHMARK_TICKER = "SPY"
BENCHMARK_DIR = Path(__file__).resolve().parent.parent / "data" / "benchmark"
TRADING_DAYS = 252


class BenchmarkError(Exception):
    pass


# ---------------------------------------------------------------- data access
def validate_price_series(rows: Sequence[Tuple[date, float]]) -> List[Tuple[date, float]]:
    """Sorted, unique, strictly positive finite closes. Raises BenchmarkError otherwise."""
    if not rows:
        raise BenchmarkError("empty benchmark series")
    seen = set()
    for d, p in rows:
        if d in seen:
            raise BenchmarkError(f"duplicate benchmark date {d}")
        seen.add(d)
        if not isinstance(p, (int, float)) or not math.isfinite(p) or p <= 0:
            raise BenchmarkError(f"invalid benchmark price on {d}: {p!r}")
    return sorted(rows, key=lambda r: r[0])


def save_benchmark(rows: Sequence[Tuple[date, float]], source: str,
                   directory: Path = BENCHMARK_DIR) -> Path:
    rows = validate_price_series(rows)
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / f"{BENCHMARK_TICKER}.csv"
    body = "date,close\n" + "\n".join(f"{d.isoformat()},{p!r}" for d, p in rows) + "\n"
    csv_path.write_text(body, encoding="utf-8")
    meta = {
        "ticker": BENCHMARK_TICKER, "source": source,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "rows": len(rows), "first_date": rows[0][0].isoformat(), "last_date": rows[-1][0].isoformat(),
        "sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
    }
    (directory / f"{BENCHMARK_TICKER}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return csv_path


def load_benchmark(directory: Path = BENCHMARK_DIR) -> Tuple[List[Tuple[date, float]], Dict[str, Any]]:
    csv_path = directory / f"{BENCHMARK_TICKER}.csv"
    meta_path = directory / f"{BENCHMARK_TICKER}.meta.json"
    if not csv_path.exists() or not meta_path.exists():
        raise BenchmarkError("SPY benchmark data not found (run scripts/fetch_spy_benchmark.py)")
    text = csv_path.read_text(encoding="utf-8")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if hashlib.sha256(text.encode("utf-8")).hexdigest() != meta.get("sha256"):
        raise BenchmarkError("SPY benchmark checksum mismatch (file modified after fetch)")
    rows = []
    for line in text.strip().split("\n")[1:]:
        d, p = line.split(",")
        rows.append((date.fromisoformat(d), float(p)))
    return validate_price_series(rows), meta


def fetch_spy(start: str, end: str, retries: int = 3, backoff: float = 2.0) -> List[Tuple[date, float]]:
    """Download SPY unadjusted closes via yfinance with bounded retry and exponential backoff."""
    import yfinance as yf  # imported lazily so tests never need the network
    last: Optional[Exception] = None
    for attempt in range(retries):
        try:
            df = yf.download(BENCHMARK_TICKER, start=start, end=end, progress=False,
                             auto_adjust=False)
            if df is None or len(df) == 0:
                raise BenchmarkError("empty response")
            close = df["Close"]
            if hasattr(close, "columns"):
                close = close[BENCHMARK_TICKER]
            return validate_price_series([(ts.date(), float(v)) for ts, v in close.dropna().items()])
        except Exception as exc:  # noqa: BLE001 - retried then re-raised below
            last = exc
            logger.warning("SPY fetch attempt %d/%d failed: %s", attempt + 1, retries, exc)
            time.sleep(backoff ** attempt)
    raise BenchmarkError(f"SPY fetch failed after {retries} attempts: {last}")


# -------------------------------------------------------------------- metrics
def _returns(series: Sequence[Tuple[date, float]]) -> List[Tuple[date, float]]:
    return [(series[i][0], series[i][1] / series[i - 1][1] - 1.0) for i in range(1, len(series))
            if series[i - 1][1] > 0]


def _max_drawdown(values: Sequence[float]) -> float:
    peak, mdd = -math.inf, 0.0
    for v in values:
        peak = max(peak, v)
        if peak > 0:
            mdd = max(mdd, (peak - v) / peak)
    return mdd


def _mean(x): return sum(x) / len(x)


def _std(x):
    if len(x) < 2:
        return 0.0
    m = _mean(x)
    return math.sqrt(sum((v - m) ** 2 for v in x) / (len(x) - 1))


def compute_performance(
    equity: Sequence[Tuple[date, float]],
    benchmark: Optional[Sequence[Tuple[date, float]]],
    transactions: Optional[Sequence[Dict[str, Any]]] = None,
    closed_pnls: Optional[Sequence[float]] = None,
    risk_free_annual: float = 0.0,
    min_obs_annualize: int = 126,
) -> Dict[str, Any]:
    """
    `equity` and `benchmark` are (date, level) series. Returns are computed on the
    common dates only. If benchmark is None/empty the result is BLOCKED.
    """
    out: Dict[str, Any] = {"benchmark_ticker": BENCHMARK_TICKER, "status": "OK", "notes": []}
    equity = sorted(equity, key=lambda r: r[0])
    if len(equity) < 2:
        return {**out, "status": "BLOCKED", "reason": "fewer than 2 equity observations"}
    if not benchmark:
        return {**out, "status": "BLOCKED",
                "reason": "valid SPY benchmark data unavailable; no substitute used"}

    bmap = {d: p for d, p in benchmark}
    emap = {d: v for d, v in equity}
    common = sorted(set(bmap) & set(emap))
    out["coverage"] = {
        "equity_dates": len(emap), "benchmark_dates": len(bmap), "aligned_dates": len(common),
        "equity_dates_without_benchmark": len(set(emap) - set(bmap)),
        "benchmark_dates_outside_equity_window": len(
            [d for d in bmap if equity[0][0] <= d <= equity[-1][0] and d not in emap]),
        "handling": "intersection of dates only; no fill or interpolation",
    }
    if len(common) < 2:
        return {**out, "status": "BLOCKED", "reason": "fewer than 2 aligned dates"}

    e = [(d, emap[d]) for d in common]
    b = [(d, bmap[d]) for d in common]
    er, br = _returns(e), _returns(b)
    n = len(er)
    cum_s = e[-1][1] / e[0][1] - 1.0
    cum_b = b[-1][1] / b[0][1] - 1.0
    daily_rf = (1 + risk_free_annual) ** (1 / TRADING_DAYS) - 1
    ex = [r - daily_rf for _, r in er]
    vol_d = _std([r for _, r in er])
    out.update({
        "window": {"start": common[0].isoformat(), "end": common[-1].isoformat(),
                   "return_observations": n},
        "cumulative_return": cum_s,
        "benchmark_return": cum_b,
        "cumulative_excess_return": cum_s - cum_b,
        "volatility_annualized": vol_d * math.sqrt(TRADING_DAYS),
        "sharpe_ratio": (_mean(ex) / vol_d * math.sqrt(TRADING_DAYS)) if vol_d > 0 else None,
        "max_drawdown": _max_drawdown([v for _, v in e]),
        "benchmark_max_drawdown": _max_drawdown([v for _, v in b]),
    })
    if n >= min_obs_annualize:
        yrs = n / TRADING_DAYS
        out["annualized_return"] = (1 + cum_s) ** (1 / yrs) - 1
        out["benchmark_annualized_return"] = (1 + cum_b) ** (1 / yrs) - 1
    else:
        out["annualized_return"] = None
        out["notes"].append(f"annualized return omitted: {n} observations < {min_obs_annualize}")

    if transactions is not None:
        traded = sum(t["shares"] * t["price"] for t in transactions)
        out["turnover_notional"] = traded
        out["turnover_vs_avg_equity"] = traded / _mean([v for _, v in e])
        out["transaction_costs_total"] = sum(t.get("fee", 0.0) for t in transactions)
        out["trade_count"] = len(transactions)
    if closed_pnls is not None:
        out["closed_trades"] = len(closed_pnls)
        out["win_rate"] = (sum(1 for p in closed_pnls if p > 0) / len(closed_pnls)) if closed_pnls else None
    out["disclaimer"] = "Historical result for this run only; not a forecast of future performance."
    return out
