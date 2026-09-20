import logging
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict, Any
from historical.analog_search import HistoricalAnalogSearch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("AnalogAgent")


class AnalogAgent:
    """
    Responsible for:
    - Querying FAISS vector index of historical situations
    - Aggregating statistical outcomes (% positive, median return, drawdown)
    - Formulating historical analog comparison report
    """
    def __init__(self):
        self.searcher = HistoricalAnalogSearch.get_instance()

    def analyze_analogs(self, ticker: str, top_k: int = 50) -> Dict[str, Any]:
        results = self.searcher.search_by_ticker(ticker, top_k=top_k)

        sim_cases = results.get("similar_cases", 0)
        sample_size = results.get("sample_size", sim_cases)
        win_rate = results.get("success_rate", 0.50)
        med_ret = results.get("median_return_5d", 0.0)
        avg_ret = results.get("avg_return_5d", 0.0)
        ci = results.get("confidence_interval", [0.0, 0.0])
        regime_dist = results.get("market_regime_distribution", {})
        sector_dist = results.get("sector_distribution", {})

        top_regime = max(regime_dist.items(), key=lambda x: x[1])[0] if regime_dist else "Neutral"

        narrative = (
            f"Identified {sim_cases} historical analog situations across 30,025 market observations. "
            f"Historical 5-day win rate is {win_rate:.1%}, with a median forward return of {med_ret:+.2%} "
            f"(mean {avg_ret:+.2%}, 95% CI [{ci[0]:+.2%}, {ci[1]:+.2%}]). "
            f"Primary analog regime cluster is {top_regime} ({regime_dist.get(top_regime, 0):.0%})."
        )

        return {
            "similar_cases": sim_cases,
            "sample_size": sample_size,
            "success_rate": win_rate,
            "median_return_5d": med_ret,
            "avg_return_5d": avg_ret,
            "confidence_interval": ci,
            "market_regime_distribution": regime_dist,
            "sector_distribution": sector_dist,
            "top_matches": results.get("top_matches", []),
            "summary": narrative,
        }


if __name__ == "__main__":
    agent = AnalogAgent()
    res = agent.analyze_analogs("NVDA")
    print("\nAnalogAgent Result for NVDA:")
    print(res["summary"])
