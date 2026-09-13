"""
Watchlist Ranking Engine for Fin_Sentiment_MLPipeline.

Ranks candidate assets using a multi-factor score:
score = signal_strength + expected_return + analog_success - risk
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict, List, Any


class WatchlistRanker:
    """Ranks assets for portfolio inclusion based on predictive signal, return, analogs, and risk."""

    def __init__(
        self,
        weight_signal: float = 1.0,
        weight_return: float = 1.0,
        weight_analog: float = 1.0,
        weight_risk: float = 1.0
    ):
        self.w_sig = weight_signal
        self.w_ret = weight_return
        self.w_ana = weight_analog
        self.w_risk = weight_risk

    def score_asset(
        self,
        ticker: str,
        signal_strength: float,
        expected_return: float,
        analog_success: float,
        risk: float
    ) -> Dict[str, Any]:
        """
        Calculates composite score for an asset.

        Args:
            ticker: Stock symbol
            signal_strength: Model prediction confidence / probability [0, 1]
            expected_return: Multi-modal expected return (e.g. 0.15)
            analog_success: Analog historical win rate [0, 1]
            risk: Asset annualized volatility or beta (e.g. 0.25)
        """
        score = (
            (self.w_sig * signal_strength) +
            (self.w_ret * expected_return) +
            (self.w_ana * analog_success) -
            (self.w_risk * risk)
        )

        return {
            "ticker": ticker,
            "score": round(score, 4),
            "signal_strength": round(signal_strength, 4),
            "expected_return": round(expected_return, 4),
            "analog_success": round(analog_success, 4),
            "risk": round(risk, 4)
        }

    def rank_universe(
        self,
        candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Scores and ranks a list of candidate assets.
        Each candidate dict should have: ticker, signal_strength, expected_return, analog_success, risk.
        Returns sorted list with ranks.
        """
        scored = []
        for c in candidates:
            res = self.score_asset(
                ticker=c["ticker"],
                signal_strength=c.get("signal_strength", 0.50),
                expected_return=c.get("expected_return", 0.10),
                analog_success=c.get("analog_success", 0.50),
                risk=c.get("risk", 0.20)
            )
            scored.append(res)

        scored.sort(key=lambda x: x["score"], reverse=True)

        ranked = []
        for idx, item in enumerate(scored, start=1):
            item["rank"] = idx
            ranked.append(item)

        return ranked


if __name__ == "__main__":
    ranker = WatchlistRanker()
    candidates = [
        {"ticker": "NVDA", "signal_strength": 0.84, "expected_return": 0.18, "analog_success": 0.78, "risk": 0.32},
        {"ticker": "MSFT", "signal_strength": 0.75, "expected_return": 0.14, "analog_success": 0.70, "risk": 0.22},
        {"ticker": "AAPL", "signal_strength": 0.65, "expected_return": 0.12, "analog_success": 0.60, "risk": 0.20},
        {"ticker": "AMD",  "signal_strength": 0.62, "expected_return": 0.13, "analog_success": 0.58, "risk": 0.34}
    ]
    ranked = ranker.rank_universe(candidates)
    print("Watchlist Ranking:")
    for r in ranked:
        print(f"{r['rank']}. {r['ticker']:<5} Score: {r['score']} (Signal: {r['signal_strength']}, ExpRet: {r['expected_return']}, Analog: {r['analog_success']}, Risk: {r['risk']})")
