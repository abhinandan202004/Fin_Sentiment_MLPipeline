import logging
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from historical.faiss_index import AnalogIndexBuilder, ANALOG_FEATURE_COLS
from config import DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("AnalogSearch")


class HistoricalAnalogSearch:
    _instance: Optional["HistoricalAnalogSearch"] = None

    def __init__(self):
        self.builder = AnalogIndexBuilder()
        self.index, self.metadata, self.scaler = self.builder.load_artifacts()

    @classmethod
    def get_instance(cls) -> "HistoricalAnalogSearch":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def find_analogs(
        self,
        query_vector: np.ndarray,
        top_k: int = 50,
        regime_filter: Optional[str] = None,
        min_cases: int = 10,
    ) -> Dict[str, Any]:
        """
        Queries FAISS for top_k nearest historical analog situations.
        Computes win rate (success_rate), median 5d return, and top matching cases.
        """
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        # Scale query vector using fitted scaler
        scaled_query = self.scaler.transform(query_vector).astype(np.float32)

        # Search in FAISS index (retrieve 2x top_k to allow filtering)
        fetch_k = min(len(self.metadata), top_k * 3)
        distances, indices = self.index.search(scaled_query, fetch_k)

        matched_cases = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx in self.metadata:
                case = self.metadata[idx].copy()
                case["distance"] = round(float(dist), 4)

                # Optional regime filter
                if regime_filter and case.get("market_regime") != regime_filter:
                    continue

                matched_cases.append(case)
                if len(matched_cases) >= top_k:
                    break

        if len(matched_cases) < min_cases and regime_filter:
            # Fallback without filter if filter was too restrictive
            return self.find_analogs(query_vector, top_k=top_k, regime_filter=None, min_cases=min_cases)

        if not matched_cases:
            return {
                "similar_cases": 0,
                "sample_size": 0,
                "success_rate": 0.50,
                "median_return_5d": 0.0,
                "avg_return_5d": 0.0,
                "confidence_interval": [0.0, 0.0],
                "market_regime_distribution": {},
                "sector_distribution": {},
                "top_matches": [],
            }

        returns_5d = [c["return_5d"] for c in matched_cases]
        n = len(returns_5d)
        success_count = sum(1 for r in returns_5d if r > 0)
        success_rate = success_count / n
        median_ret = float(np.median(returns_5d))
        avg_ret = float(np.mean(returns_5d))
        std_ret = float(np.std(returns_5d, ddof=1)) if n > 1 else 0.0

        # 95% Confidence Interval for mean 5-day return
        se = std_ret / np.sqrt(n) if n > 0 else 0.0
        ci_lower = round(avg_ret - 1.96 * se, 4)
        ci_upper = round(avg_ret + 1.96 * se, 4)

        # Distribution across Market Regimes
        regime_counts = {}
        for c in matched_cases:
            reg = c.get("market_regime", "Neutral")
            regime_counts[reg] = regime_counts.get(reg, 0) + 1
        regime_dist = {k: round(v / n, 3) for k, v in regime_counts.items()}

        # Distribution across Sectors
        sector_counts = {}
        for c in matched_cases:
            sec = c.get("sector", "Other")
            sector_counts[sec] = sector_counts.get(sec, 0) + 1
        sector_dist = {k: round(v / n, 3) for k, v in sector_counts.items()}

        return {
            "similar_cases": n,
            "sample_size": n,
            "success_rate": round(float(success_rate), 4),
            "median_return_5d": round(float(median_ret), 4),
            "avg_return_5d": round(float(avg_ret), 4),
            "confidence_interval": [ci_lower, ci_upper],
            "market_regime_distribution": regime_dist,
            "sector_distribution": sector_dist,
            "top_matches": matched_cases[:5],
        }

    def search_by_ticker(self, ticker: str, top_k: int = 50) -> Dict[str, Any]:
        """
        Constructs the latest 10-feature vector for ticker and runs analog search.
        """
        from models.feature_store import load_feature_dataset
        df = load_feature_dataset()
        ticker_df = df[df["ticker"] == ticker.upper()].copy()
        if ticker_df.empty:
            raise ValueError(f"No historical records found for ticker {ticker}")

        latest_row = ticker_df.sort_values(by="date").iloc[-1]

        # Extract 10 analog features exactly matching ANALOG_FEATURE_COLS
        vec_values = []
        for col in ANALOG_FEATURE_COLS:
            val = latest_row.get(col, 0.0)
            if pd.isna(val):
                val = 0.0
            vec_values.append(float(val))

        vec = np.array([vec_values], dtype=np.float32)
        return self.find_analogs(vec, top_k=top_k)


if __name__ == "__main__":
    searcher = HistoricalAnalogSearch.get_instance()
    for test_t in ["NVDA", "AAPL", "JPM", "XOM"]:
        res = searcher.search_by_ticker(test_t, top_k=50)
        print(f"\n--- Historical Analog Search: {test_t} ---")
        print(f"Sample Size: {res['sample_size']} cases | Win Rate: {res['success_rate']:.1%}")
        print(f"Median 5d: {res['median_return_5d']:+.2%} | Mean 5d: {res['avg_return_5d']:+.2%} | 95% CI: [{res['confidence_interval'][0]:+.2%}, {res['confidence_interval'][1]:+.2%}]")
        print(f"Regime Dist: {res['market_regime_distribution']}")
        print(f"Sector Dist: {res['sector_distribution']}")
        top1 = res['top_matches'][0]
        print(f"Top Precedent: {top1['date']} ({top1['ticker']}, {top1.get('sector', '')}) 5d: {top1['return_5d']:+.2%} dist: {top1['distance']}")
