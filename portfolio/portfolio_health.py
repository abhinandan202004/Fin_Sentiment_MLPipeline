"""
Portfolio Health Score Engine for Fin_Sentiment_MLPipeline.

Computes a holistic 0-100 score and letter grade based on:
1. Diversification (HHI and asset spread)
2. Risk profile (volatility and portfolio beta)
3. Cash buffer (adequate liquidity without drag)
4. Drawdown risk (historical / simulated maximum drawdown)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict, Any


class PortfolioHealthEngine:
    """Evaluates portfolio health and assigns a letter grade."""

    def compute_health_score(
        self,
        hhi: float = 0.21,
        volatility: float = 0.18,
        beta: float = 1.12,
        cash_buffer: float = 0.10,
        max_drawdown: float = 0.12
    ) -> Dict[str, Any]:
        """
        Computes composite health score:
        health_score = diversification + risk_score + cash_buffer + drawdown_score
        """
        # 1. Diversification Score (0 - 25 points)
        if hhi <= 0.18:
            div_score = 25.0
        elif hhi <= 0.25:
            div_score = 22.0
        elif hhi <= 0.35:
            div_score = 17.0
        elif hhi <= 0.50:
            div_score = 12.0
        else:
            div_score = 7.0

        # 2. Risk Score (0 - 25 points)
        risk_score = 25.0
        if volatility > 0.25:
            risk_score -= 10.0
        elif volatility > 0.18:
            risk_score -= 5.0

        if beta > 1.30:
            risk_score -= 6.0
        elif beta > 1.15:
            risk_score -= 3.0
        risk_score = max(risk_score, 5.0)

        # 3. Cash Buffer Score (0 - 25 points)
        if 0.08 <= cash_buffer <= 0.20:
            cash_score = 25.0
        elif 0.05 <= cash_buffer < 0.08 or 0.20 < cash_buffer <= 0.30:
            cash_score = 20.0
        elif cash_buffer < 0.05:
            cash_score = 12.0  # Illiquidity risk
        else:
            cash_score = 15.0  # Excessive cash drag

        # 4. Drawdown Score (0 - 25 points)
        if max_drawdown <= 0.08:
            dd_score = 25.0
        elif max_drawdown <= 0.14:
            dd_score = 21.0
        elif max_drawdown <= 0.22:
            dd_score = 15.0
        else:
            dd_score = 8.0

        total_score = int(round(div_score + risk_score + cash_score + dd_score))
        total_score = min(max(total_score, 0), 100)

        grade = self._map_grade(total_score)

        return {
            "portfolio_health": total_score,
            "grade": grade,
            "breakdown": {
                "diversification": round(div_score, 1),
                "risk_score": round(risk_score, 1),
                "cash_buffer": round(cash_score, 1),
                "drawdown_score": round(dd_score, 1)
            }
        }

    def _map_grade(self, score: int) -> str:
        if score >= 90:
            return "A+"
        elif score >= 85:
            return "A"
        elif score >= 80:
            return "A-"
        elif score >= 75:
            return "B+"
        elif score >= 70:
            return "B"
        elif score >= 65:
            return "B-"
        elif score >= 60:
            return "C+"
        elif score >= 50:
            return "C"
        else:
            return "D"


if __name__ == "__main__":
    health_engine = PortfolioHealthEngine()
    result = health_engine.compute_health_score(
        hhi=0.21,
        volatility=0.18,
        beta=1.12,
        cash_buffer=0.10,
        max_drawdown=0.12
    )
    print("Portfolio Health Evaluation:")
    print(f"Health Score: {result['portfolio_health']}")
    print(f"Grade:        {result['grade']}")
    print(f"Breakdown:    {result['breakdown']}")
