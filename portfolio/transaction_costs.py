from dataclasses import dataclass
from typing import Dict
import numpy as np


@dataclass
class TransactionCostModel:
    """
    Transaction cost model accounting for:
    - Brokerage commission (e.g. 0.03%)
    - Market impact / slippage (e.g. 0.05%)
    - Exchange fees & taxes (e.g. 0.02%)
    Default total roundtrip friction: 10 bps (0.001).
    """
    cost_rate: float = 0.001

    def compute_rebalance_cost(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        total_capital: float,
    ) -> float:
        """
        Calculates total expected transaction cost in currency units
        based on turnover between current and target weights.
        """
        all_assets = set(current_weights.keys()).union(set(target_weights.keys()))
        turnover = 0.0
        for asset in all_assets:
            if asset.lower() == "cash":
                continue
            curr_w = current_weights.get(asset, 0.0)
            tgt_w = target_weights.get(asset, 0.0)
            turnover += abs(tgt_w - curr_w)

        # Turnover represents two-sided buy/sell reallocation
        cost_dollars = turnover * total_capital * self.cost_rate
        return cost_dollars
