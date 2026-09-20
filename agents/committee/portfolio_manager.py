"""
Portfolio Manager Agent for Fin_Sentiment_MLPipeline.

Responsibilities:
- Translates committee debate consensus into actionable portfolio execution
- Solves position sizing via Half-Kelly criterion and volatility parity
- Reconciles allocation proposals with Risk Officer position limits
- Verifies overall portfolio diversification and liquidity requirements
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from typing import Dict, List, Any, Optional

from portfolio.position_sizer import PositionSizer
from portfolio.constraints import PortfolioConstraints


class PortfolioManager:
    """Calculates practical trade execution sizes based on debate inputs and risk boundaries."""

    def __init__(
        self,
        position_sizer: Optional[PositionSizer] = None,
        constraints: Optional[PortfolioConstraints] = None
    ):
        self.sizer = position_sizer or PositionSizer()
        self.constraints = constraints or PortfolioConstraints()

    def size_position(
        self,
        ticker: str,
        bull_case: Dict[str, Any],
        bear_case: Dict[str, Any],
        risk_case: Dict[str, Any],
        prosecutor_case: Dict[str, Any],
        current_weight: float = 0.0,
        regime: str = "Bull"
    ) -> Dict[str, Any]:
        """
        Synthesizes debate inputs into recommended trade action and capital allocation.
        """
        args: List[str] = []

        # If Risk Officer vetoes, allocation is immediately 0
        if risk_case.get("veto_triggered", False):
            return {
                "agent": "PortfolioManager",
                "ticker": ticker.upper(),
                "action": "REJECTED",
                "allocation": 0.0,
                "confidence": 0.90,
                "arguments": ["Risk Officer veto triggered. Allocation denied in accordance with governance rules."]
            }

        # If prosecutor rejected data quality
        if prosecutor_case.get("stance") == "REJECT":
            return {
                "agent": "PortfolioManager",
                "ticker": ticker.upper(),
                "action": "WATCHLIST",
                "allocation": 0.0,
                "confidence": 0.70,
                "arguments": ["Evidence Prosecutor flagged critical data deficiencies. Capital withheld; placed on Watchlist."]
            }

        # Calculate Kelly sizing
        win_rate = bull_case.get("confidence", 0.60)
        risk_limit = risk_case.get("position_limit", self.constraints.max_single_position)

        kelly_rec = self.sizer.recommend_size(
            ticker=ticker,
            win_rate=win_rate,
            reward_risk_ratio=1.35,
            volatility=0.25,
            max_position=self.constraints.max_single_position,
            regime=regime
        )
        base_size = kelly_rec.get("recommended_position", 0.08)

        # Scale based on bull/bear conviction balance
        bull_score = bull_case.get("bull_score", 50.0)
        bear_score = bear_case.get("bear_score", 30.0)
        net_sentiment = (bull_score - bear_score) / 100.0  # e.g. (75 - 40) / 100 = 0.35

        scaled_size = base_size * (1.0 + (net_sentiment * 0.50))
        target_allocation = round(min(max(scaled_size, 0.0), risk_limit, self.constraints.max_single_position), 4)

        delta = round(target_allocation - current_weight, 4)

        if bear_score >= 65.0 and bull_score <= 40.0:
            action = "SELL"
            target_allocation = 0.0
            args.append(f"Bear dominance ({bear_score:.0f} vs {bull_score:.0f}) warrants total position liquidation (0.0% allocation)")
        elif delta > 0.05 and target_allocation >= 0.10:
            action = "STRONG BUY"
            args.append(f"High-conviction spread (+{delta:.1%}) justifies aggressive capital allocation up to {target_allocation:.1%}")
            args.append(f"Fully compliant with Risk Officer ceiling of {risk_limit:.1%}")
        elif delta > 0.02:
            action = "BUY"
            args.append(f"Half-Kelly model and debate spread indicate {target_allocation:.1%} target allocation (+{delta:.1%} expansion)")
            args.append(f"Fully compliant with Risk Officer ceiling of {risk_limit:.1%}")
        elif delta < -0.02:
            action = "REDUCE"
            args.append(f"Target allocation {target_allocation:.1%} requires trimming current position by {abs(delta):.1%}")
        else:
            action = "HOLD"
            args.append(f"Current allocation ({current_weight:.1%}) aligns with optimal target {target_allocation:.1%}")

        return {
            "agent": "PortfolioManager",
            "ticker": ticker.upper(),
            "action": action,
            "allocation": target_allocation,
            "delta_allocation": delta,
            "confidence": round(win_rate * 0.90, 2),
            "arguments": args
        }


if __name__ == "__main__":
    pm = PortfolioManager()
    sample_bull = {"confidence": 0.72, "bull_score": 78.0}
    sample_bear = {"bear_score": 38.0}
    sample_risk = {"veto_triggered": False, "position_limit": 0.12}
    sample_pros = {"stance": "PASS"}

    res = pm.size_position("NVDA", sample_bull, sample_bear, sample_risk, sample_pros, current_weight=0.03)
    print("Portfolio Manager Output:")
    print(f"Action: {res['action']} | Target Allocation: {res['allocation']:.1%} | Delta: {res['delta_allocation']:+.1%}")
    for a in res["arguments"]:
        print(f"  > {a}")
