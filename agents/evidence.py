from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional


@dataclass
class Evidence:
    ticker: str

    prediction: str = "NEUTRAL"
    confidence: float = 0.50

    events: List[Dict[str, Any]] = field(default_factory=list)
    graph_impacts: List[str] = field(default_factory=list)

    analogs: Dict[str, Any] = field(default_factory=dict)

    technical_summary: Dict[str, Any] = field(default_factory=dict)
    risk_summary: Dict[str, Any] = field(default_factory=dict)

    shap_drivers: List[str] = field(default_factory=list)
    feature_contributions: List[Dict[str, Any]] = field(default_factory=list)

    # Sprint 4.1 Enhanced Evidence Contract
    raw_probability: float = 0.50
    optimal_threshold: float = 0.55
    market_regime: str = "Neutral"
    quant_features: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Converts Evidence object to a clean dictionary."""
        return asdict(self)
