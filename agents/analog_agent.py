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

    def analyze_analogs(self, ticker: str, top_k: int = 42) -> Dict[str, Any]:
        results = self.searcher.search_by_ticker(ticker, top_k=top_k)

        sim_cases = results["similar_cases"]
        win_rate = results["success_rate"]
        med_ret = results["median_return_5d"]
        avg_ret = results["avg_return_5d"]

        narrative = (
            f"Identified {sim_cases} highly correlated historical analogs based on multi-modal feature vector matching. "
            f"Across these historical setups, {win_rate:.1%} of cases yielded positive 5-day forward returns, "
            f"with a median forward return of {med_ret:+.2%} and an average return of {avg_ret:+.2%}."
        )

        return {
            "similar_cases": sim_cases,
            "success_rate": win_rate,
            "median_return_5d": med_ret,
            "avg_return_5d": avg_ret,
            "top_matches": results["top_matches"],
            "summary": narrative,
        }


if __name__ == "__main__":
    agent = AnalogAgent()
    res = agent.analyze_analogs("NVDA")
    print("\nAnalogAgent Result for NVDA:")
    print(res["summary"])
