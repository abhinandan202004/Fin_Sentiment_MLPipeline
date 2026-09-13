from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class AnalogCase:
    vector_id: int
    date: str
    ticker: str
    event: str
    return_1d: float
    return_5d: float
    return_10d: float
    sentiment: float
    rsi: float
    volume_ratio: float
    market_regime: str  # "low_vol", "high_vol", "bull_trend", "bear_trend"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
