"""
Governance Rules & Risk Veto Engine for Fin_Sentiment_MLPipeline.

Enforces strict portfolio guardrails and institutional veto rules:
- MAX_SINGLE_POSITION: 30%
- MAX_SECTOR_EXPOSURE: 40%
- MIN_CASH_BUFFER: 10%
- MAX_PORTFOLIO_BETA: 1.20
- MAX_VAR_95: 4.0%

Defines RISK_VETO_RULES that can reject or constrain investment committee proposals.
"""

from typing import Dict, List, Any, Optional

MAX_SINGLE_POSITION = 0.30
MAX_SECTOR_EXPOSURE = 0.40
MIN_CASH_BUFFER = 0.10
MAX_PORTFOLIO_BETA = 1.20
MAX_VAR_95 = 0.040

RISK_VETO_RULES = [
    "portfolio_var_exceeded",
    "max_sector_exposure_exceeded",
    "portfolio_beta_exceeded",
    "liquidity_risk_high"
]


class GovernanceEngine:
    """Evaluates proposed allocations against compliance rules and risk veto triggers."""

    def __init__(
        self,
        max_single_position: float = MAX_SINGLE_POSITION,
        max_sector_exposure: float = MAX_SECTOR_EXPOSURE,
        min_cash_buffer: float = MIN_CASH_BUFFER,
        max_portfolio_beta: float = MAX_PORTFOLIO_BETA,
        max_var_95: float = MAX_VAR_95
    ):
        self.max_single_position = max_single_position
        self.max_sector_exposure = max_sector_exposure
        self.min_cash_buffer = min_cash_buffer
        self.max_portfolio_beta = max_portfolio_beta
        self.max_var_95 = max_var_95

    def evaluate_compliance(
        self,
        ticker: str,
        target_allocation: float,
        current_portfolio: Optional[Dict[str, float]] = None,
        sector_mapping: Optional[Dict[str, str]] = None,
        risk_metrics: Optional[Dict[str, Any]] = None,
        cash_buffer: float = 0.10
    ) -> Dict[str, Any]:
        """
        Runs comprehensive governance checks.
        Returns compliance status, list of triggered vetoes, and detailed rationale.
        """
        current_portfolio = current_portfolio or {}
        sector_mapping = sector_mapping or {}
        risk_metrics = risk_metrics or {}

        vetoes: List[str] = []
        violations: List[str] = []

        # 1. Single Position Cap Check
        if target_allocation > self.max_single_position:
            violations.append(
                f"Proposed allocation {target_allocation:.1%} exceeds maximum single position limit of {self.max_single_position:.1%}"
            )

        # 2. Sector Exposure Check
        ticker_sector = sector_mapping.get(ticker, "Other")
        current_sector_exp = 0.0
        for sym, weight in current_portfolio.items():
            if sym != "Cash" and sector_mapping.get(sym, "Other") == ticker_sector:
                current_sector_exp += weight

        projected_sector_exp = current_sector_exp + max(0.0, target_allocation - current_portfolio.get(ticker, 0.0))
        if projected_sector_exp > self.max_sector_exposure:
            vetoes.append("max_sector_exposure_exceeded")
            violations.append(
                f"Projected sector exposure for '{ticker_sector}' ({projected_sector_exp:.1%}) exceeds limit of {self.max_sector_exposure:.1%}"
            )

        # 3. Minimum Cash Buffer Check
        if cash_buffer < self.min_cash_buffer:
            vetoes.append("liquidity_risk_high")
            violations.append(
                f"Portfolio cash buffer ({cash_buffer:.1%}) is below mandatory threshold of {self.min_cash_buffer:.1%}"
            )

        # 4. Portfolio Beta Check
        proj_beta = risk_metrics.get("beta", 1.0)
        if proj_beta > self.max_portfolio_beta:
            vetoes.append("portfolio_beta_exceeded")
            violations.append(
                f"Projected portfolio beta ({proj_beta:.2f}) exceeds risk ceiling of {self.max_portfolio_beta:.2f}"
            )

        # 5. Portfolio VaR Check
        proj_var = risk_metrics.get("var_95", 0.025)
        if proj_var > self.max_var_95:
            vetoes.append("portfolio_var_exceeded")
            violations.append(
                f"Projected 1-day 95% VaR ({proj_var:.2%}) exceeds risk tolerance of {self.max_var_95:.2%}"
            )

        passed = len(vetoes) == 0 and len(violations) == 0

        return {
            "governance_passed": passed,
            "vetoes_triggered": vetoes,
            "violations": violations,
            "capped_allocation": min(target_allocation, self.max_single_position) if passed else 0.0,
            "sector": ticker_sector,
            "reason": "; ".join(violations) if violations else "All governance and risk rules passed."
        }


if __name__ == "__main__":
    gov = GovernanceEngine()
    print("--- 1. Testing Compliant Allocation ---")
    res1 = gov.evaluate_compliance(
        ticker="NVDA",
        target_allocation=0.15,
        current_portfolio={"NVDA": 0.05, "AAPL": 0.20, "Cash": 0.15},
        sector_mapping={"NVDA": "Technology", "AAPL": "Technology"},
        risk_metrics={"beta": 1.05, "var_95": 0.028},
        cash_buffer=0.15
    )
    print("Passed:", res1["governance_passed"], "| Reason:", res1["reason"])

    print("\n--- 2. Testing Veto: Sector Exposure Exceeded ---")
    res2 = gov.evaluate_compliance(
        ticker="NVDA",
        target_allocation=0.30,
        current_portfolio={"MSFT": 0.25, "AAPL": 0.20, "Cash": 0.10},
        sector_mapping={"NVDA": "Technology", "MSFT": "Technology", "AAPL": "Technology"},
        risk_metrics={"beta": 1.10, "var_95": 0.031},
        cash_buffer=0.10
    )
    print("Passed:", res2["governance_passed"], "| Vetoes:", res2["vetoes_triggered"], "| Reason:", res2["reason"])
