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
    sector: str = "Technology"
    vix_level: float = 20.0
    sentiment_rank: float = 0.50
    sector_return_5d: float = 0.0
    bb_squeeze: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
