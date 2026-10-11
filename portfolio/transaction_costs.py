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
    commission_rate: float = 0.0005   # 5 bps per side
    slippage_rate: float = 0.0002     # 2 bps per side
    spread_rate: float = 0.0003       # 3 bps per side

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

    def compute_trade_cost(self, trade_value: float, side: str = "BUY") -> Dict[str, float]:
        """
        Deducts transaction costs once per applicable side.
        Friction breakdown:
        - Commission: 5 bps (0.0005)
        - Slippage: 2 bps (0.0002)
        - Spread / Exchange fees: 3 bps (0.0003)
        Total one-way: 10 bps (0.0010)
        """
        comm = trade_value * self.commission_rate
        slip = trade_value * self.slippage_rate
        spread = trade_value * self.spread_rate
        total = comm + slip + spread
        return {
            "commission": comm,
            "slippage": slip,
            "spread": spread,
            "total_cost": total,
        }
