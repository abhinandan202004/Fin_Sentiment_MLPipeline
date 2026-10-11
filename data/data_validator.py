"""
Market Data Quality & Reliability Validator for Sprint 10.
Production Readiness, Risk Monitoring & Paper-Trading Reliability.

Validates:
1. Price bar sanity: non-zero, positive close/open, high >= low, volume >= 0, jump bounds.
2. Freshness & staleness: detects missing trading sessions while honoring weekends and market holidays.
3. Feature readiness: ensures complete 43-feature vectors before ML model inference.
4. Operational Telemetry: tracks ingestion freshness and anomaly statistics.
"""

import logging
from datetime import date, datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple

logger = logging.getLogger("DataValidator")

# US Stock Market Holidays (fixed & observed standard dates)
US_MARKET_HOLIDAYS_YEARS = {
    # 2024
    date(2024, 1, 1), date(2024, 1, 15), date(2024, 2, 19), date(2024, 3, 29),
    date(2024, 5, 27), date(2024, 6, 19), date(2024, 7, 4), date(2024, 9, 2),
    date(2024, 11, 28), date(2024, 12, 25),
    # 2025
    date(2025, 1, 1), date(2025, 1, 20), date(2025, 2, 17), date(2025, 4, 18),
    date(2025, 5, 26), date(2025, 6, 19), date(2025, 7, 4), date(2025, 9, 1),
    date(2025, 11, 27), date(2025, 12, 25),
    # 2026
    date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3),
    date(2026, 5, 25), date(2026, 6, 19), date(2026, 7, 3), date(2026, 9, 7),
    date(2026, 11, 26), date(2026, 12, 25),
}


class DataValidator:
    """Institutional market data validation and freshness monitor."""

    def __init__(self, holidays: Optional[Set[date]] = None):
        self.holidays = holidays or US_MARKET_HOLIDAYS_YEARS
        self.stats = {
            "bars_validated": 0,
            "bars_passed": 0,
            "bars_failed": 0,
            "stale_detections": 0,
            "feature_checks_passed": 0,
            "feature_checks_failed": 0,
            "anomalies": []
        }

    def is_market_day(self, d: date) -> bool:
        """Returns True if the date is a regular weekday and not a designated holiday."""
        if d.weekday() >= 5:  # Saturday (5) or Sunday (6)
            return False
        if d in self.holidays:
            return False
        return True

    def count_expected_market_sessions(self, start_date: date, end_date: date) -> int:
        """Counts how many official market sessions occurred between start_date and end_date exclusive."""
        if start_date >= end_date:
            return 0
        sessions = 0
        cur = start_date + timedelta(days=1)
        while cur <= end_date:
            if self.is_market_day(cur):
                sessions += 1
            cur += timedelta(days=1)
        return sessions

    def check_staleness(
        self,
        bar_date: date,
        current_date: date,
        max_allowed_missed_sessions: int = 2
    ) -> Tuple[bool, int, str]:
        """
        Determines whether a price bar is stale relative to current_date.
        Returns: (is_stale, missed_sessions_count, reason_message)
        """
        if isinstance(bar_date, datetime):
            bar_date = bar_date.date()
        if isinstance(current_date, datetime):
            current_date = current_date.date()

        if bar_date > current_date:
            return False, 0, "Future bar timestamp"

        missed = self.count_expected_market_sessions(bar_date, current_date)
        if missed > max_allowed_missed_sessions:
            self.stats["stale_detections"] += 1
            msg = f"Data is stale: {missed} trading sessions elapsed since {bar_date.isoformat()} (max allowed: {max_allowed_missed_sessions})"
            return True, missed, msg

        return False, missed, "Fresh data"

    def validate_price_bar(self, bar: Any, ticker: str = "") -> Tuple[bool, List[str]]:
        """
        Validates individual price bar integrity:
        - Price must be positive (> 0.0)
        - High >= Low
        - High >= Open, High >= Close
        - Low <= Open, Low <= Close
        - Volume >= 0
        - Price jump bounds (daily return between -50% and +100%)
        """
        self.stats["bars_validated"] += 1
        errors = []

        # Extract bar attributes supporting either dict or SQLAlchemy MarketData object
        if isinstance(bar, dict):
            o = float(bar.get("open", 0.0) or 0.0)
            h = float(bar.get("high", 0.0) or 0.0)
            l = float(bar.get("low", 0.0) or 0.0)
            c = float(bar.get("close", 0.0) or 0.0)
            v = float(bar.get("volume", 0.0) or 0.0)
        else:
            o = float(getattr(bar, "open", 0.0) or 0.0)
            h = float(getattr(bar, "high", 0.0) or 0.0)
            l = float(getattr(bar, "low", 0.0) or 0.0)
            c = float(getattr(bar, "close", 0.0) or 0.0)
            v = float(getattr(bar, "volume", 0.0) or 0.0)

        if c <= 0.0 or o <= 0.0 or h <= 0.0 or l <= 0.0:
            errors.append(f"Non-positive price in bar: open={o}, high={h}, low={l}, close={c}")

        if h < l:
            errors.append(f"Invalid bar range: high ({h}) < low ({l})")

        if h < max(o, c) - 1e-4:
            errors.append(f"High ({h}) is less than max(open, close) ({max(o, c)})")

        if l > min(o, c) + 1e-4:
            errors.append(f"Low ({l}) is greater than min(open, close) ({min(o, c)})")

        if v < 0:
            errors.append(f"Negative trading volume: {v}")

        # Check extreme price jump anomaly
        if o > 0.0:
            intraday_change = (c - o) / o
            if intraday_change > 1.00 or intraday_change < -0.50:
                errors.append(f"Extreme price jump detected: {intraday_change:+.1%}")

        if errors:
            self.stats["bars_failed"] += 1
            self.stats["anomalies"].append({"ticker": ticker, "errors": errors})
            return False, errors

        self.stats["bars_passed"] += 1
        return True, []

    def validate_features_for_inference(
        self,
        features: Dict[str, Any],
        required_features: List[str]
    ) -> Tuple[bool, List[str]]:
        """
        Validates that a feature vector is completely populated before ML model inference.
        Rejects missing keys, None values, and NaN/inf values.
        """
        errors = []
        for feat in required_features:
            if feat not in features:
                errors.append(f"Missing required feature: {feat}")
                continue
            val = features[feat]
            if val is None:
                errors.append(f"Feature '{feat}' is None")
            elif isinstance(val, float) and (val != val or val in (float("inf"), float("-inf"))):
                errors.append(f"Feature '{feat}' contains NaN or Inf: {val}")

        if errors:
            self.stats["feature_checks_failed"] += 1
            return False, errors

        self.stats["feature_checks_passed"] += 1
        return True, []

    def get_telemetry(self) -> Dict[str, Any]:
        """Returns structured data validation telemetry metrics."""
        return {
            "total_bars_validated": self.stats["bars_validated"],
            "bars_passed": self.stats["bars_passed"],
            "bars_failed": self.stats["bars_failed"],
            "pass_rate": (self.stats["bars_passed"] / self.stats["bars_validated"]) if self.stats["bars_validated"] > 0 else 1.0,
            "stale_detections": self.stats["stale_detections"],
            "feature_checks_passed": self.stats["feature_checks_passed"],
            "feature_checks_failed": self.stats["feature_checks_failed"],
            "recent_anomalies_count": len(self.stats["anomalies"])
        }
