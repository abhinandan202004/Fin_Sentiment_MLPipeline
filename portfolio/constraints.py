from dataclasses import dataclass, asdict
from typing import Dict, Any


@dataclass
class PortfolioConstraints:
    """
    Core institutional investment constraints:
    - max_single_position: Maximum allocation in any single stock (e.g., 30%)
    - max_sector_exposure: Maximum allocation in any single sector (e.g., 40%)
    - minimum_cash: Minimum cash reserve allocation (e.g., 10%)
    - max_portfolio_beta: Maximum overall weighted portfolio beta vs benchmark (e.g., 1.20)
    """
    max_single_position: float = 0.30
    max_sector_exposure: float = 0.40
    minimum_cash: float = 0.10
    max_portfolio_beta: float = 1.20

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
