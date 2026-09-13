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
                "success_rate": 0.50,
                "median_return_5d": 0.0,
                "avg_return_5d": 0.0,
                "top_matches": [],
            }

        returns_5d = [c["return_5d"] for c in matched_cases]
        success_count = sum(1 for r in returns_5d if r > 0)
        success_rate = success_count / len(returns_5d)
        median_ret = float(np.median(returns_5d))
        avg_ret = float(np.mean(returns_5d))

        return {
            "similar_cases": len(matched_cases),
            "success_rate": round(float(success_rate), 4),
            "median_return_5d": round(float(median_ret), 4),
            "avg_return_5d": round(float(avg_ret), 4),
            "top_matches": matched_cases[:5],
        }

    def search_by_ticker(self, ticker: str, top_k: int = 40) -> Dict[str, Any]:
        """
        Constructs the latest feature vector for ticker and runs analog search.
        """
        csv_path = DATA_DIR / "training_dataset.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Training dataset not found at {csv_path}")

        df = pd.read_csv(csv_path)
        ticker_df = df[df["ticker"] == ticker].copy()
        if ticker_df.empty:
            raise ValueError(f"No historical records found for ticker {ticker}")

        latest_row = ticker_df.sort_values(by="date").iloc[-1]

        # Extract 7 analog features
        vec = np.array([[
            float(latest_row.get("avg_sentiment", 0.0)),
            float(latest_row.get("rsi", 50.0)),
            float(latest_row.get("macd", 0.0)),
            float(latest_row.get("volume_ratio", 1.0)),
            float(latest_row.get("sentiment_ema_3", 0.0) * 0.8),
            float(latest_row.get("spy_volatility", 0.15)),
            float(latest_row.get("spy_return_5d", 0.0)),
        ]])

        return self.find_analogs(vec, top_k=top_k)


if __name__ == "__main__":
    searcher = HistoricalAnalogSearch.get_instance()
    res = searcher.search_by_ticker("NVDA", top_k=42)
    print("\nNVDA Historical Analog Search Results:")
    print(f"Similar cases found : {res['similar_cases']}")
    print(f"Success rate (5d >0): {res['success_rate']:.2%}")
    print(f"Median 5-day return : {res['median_return_5d']:+.2%}")
    print(f"Average 5-day return: {res['avg_return_5d']:+.2%}")
    print("\nTop 3 historical case dates:")
    for m in res["top_matches"][:3]:
        print(f" - {m['date']} ({m['ticker']}): 5d Return = {m['return_5d']:+.2%}, Event = '{m['event']}', Distance = {m['distance']}")
