"""
Portfolio Optimizer for Fin_Sentiment_MLPipeline.

Uses scipy.optimize to solve:
1. Maximum Sharpe Ratio
2. Minimum Variance
3. Risk Parity (Equal Risk Contribution)

Subject to:
- Full investment (weights + cash = 1.0)
- Single asset bounds [0, max_single_position]
- Minimum cash buffer [minimum_cash, 1.0]
- Sector exposure caps
- Turnover / transaction cost friction
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, List, Optional, Tuple, Any

from portfolio.constraints import PortfolioConstraints
from portfolio.transaction_costs import TransactionCostModel


class PortfolioOptimizer:
    """Quantitative portfolio optimizer using Modern Portfolio Theory and Risk Parity."""

    def __init__(
        self,
        risk_free_rate: float = 0.04,
        constraints: Optional[PortfolioConstraints] = None,
        cost_model: Optional[TransactionCostModel] = None
    ):
        self.rf = risk_free_rate
        self.constraints = constraints or PortfolioConstraints()
        self.cost_model = cost_model or TransactionCostModel()

    def _portfolio_performance(
        self,
        weights: np.ndarray,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray
    ) -> Tuple[float, float, float]:
        """Calculates expected return, volatility, and Sharpe ratio."""
        p_return = float(np.sum(weights * expected_returns))
        p_vol = float(np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))))
        p_vol = max(p_vol, 1e-6)
        sharpe = (p_return - self.rf) / p_vol
        return p_return, p_vol, sharpe

    def optimize(
        self,
        expected_returns: Dict[str, float],
        cov_matrix: pd.DataFrame,
        mode: str = "max_sharpe",
        cash_weight: Optional[float] = None,
        current_weights: Optional[Dict[str, float]] = None,
        sector_mapping: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Optimizes asset weights under user-specified mode and constraints.

        Args:
            expected_returns: Dict of {ticker: expected_annual_return}
            cov_matrix: Annualized covariance DataFrame (tickers x tickers)
            mode: 'max_sharpe', 'min_variance', or 'risk_parity'
            cash_weight: Pre-allocated cash fraction (e.g. from RegimeAgent). If None, minimum_cash is used.
            current_weights: Existing portfolio weights {ticker: weight} for turnover penalty
            sector_mapping: {ticker: sector_name} to enforce max_sector_exposure
        """
        tickers = [t for t in expected_returns.keys() if t in cov_matrix.index]
        n_assets = len(tickers)

        if n_assets == 0:
            return {"weights": {"Cash": 1.0}, "metrics": {"return": self.rf, "volatility": 0.0, "sharpe": 0.0}}

        # Cash allocation
        target_cash = max(
            cash_weight if cash_weight is not None else self.constraints.minimum_cash,
            self.constraints.minimum_cash
        )
        equity_budget = max(1.0 - target_cash, 0.05)

        # Align inputs
        mu = np.array([expected_returns[t] for t in tickers])
        sub_cov = cov_matrix.loc[tickers, tickers].values

        # Ensure covariance matrix is positive semi-definite
        min_eig = np.min(np.real(np.linalg.eigvals(sub_cov)))
        if min_eig < 1e-6:
            sub_cov += np.eye(n_assets) * (1e-6 - min_eig + 1e-4)

        # Bounds per asset: [0, max_single_position]
        max_pos = min(self.constraints.max_single_position, equity_budget)
        bounds = tuple((0.0, max_pos) for _ in range(n_assets))

        # Constraints
        # 1. Sum of equity weights == equity_budget
        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - equity_budget}]

        # 2. Sector constraints
        if sector_mapping and self.constraints.max_sector_exposure < 1.0:
            sectors = set(sector_mapping.get(t, "Other") for t in tickers)
            for sec in sectors:
                sec_indices = [i for i, t in enumerate(tickers) if sector_mapping.get(t, "Other") == sec]
                if sec_indices:
                    cons.append({
                        'type': 'ineq',
                        'fun': lambda w, idxs=sec_indices: self.constraints.max_sector_exposure - np.sum(w[idxs])
                    })

        # Initial guess: equal weight across equity budget
        w0 = np.full(n_assets, equity_budget / n_assets)

        # Objective functions
        if mode == "max_sharpe":
            def objective(w):
                ret, vol, _ = self._portfolio_performance(w, mu, sub_cov)
                # Combine risk-free rate for the cash portion
                total_ret = ret + target_cash * self.rf
                sharpe = (total_ret - self.rf) / vol
                obj = -sharpe
                # Add turnover cost penalty if current_weights provided
                if current_weights:
                    curr_vec = np.array([current_weights.get(t, 0.0) for t in tickers])
                    turnover = np.sum(np.abs(w - curr_vec))
                    obj += self.cost_model.cost_rate * turnover * 10
                return obj

        elif mode == "min_variance":
            def objective(w):
                _, vol, _ = self._portfolio_performance(w, mu, sub_cov)
                obj = vol
                if current_weights:
                    curr_vec = np.array([current_weights.get(t, 0.0) for t in tickers])
                    turnover = np.sum(np.abs(w - curr_vec))
                    obj += self.cost_model.cost_rate * turnover * 5
                return obj

        elif mode == "risk_parity":
            # Equal Risk Contribution
            def objective(w):
                # Total portfolio variance
                p_var = np.dot(w.T, np.dot(sub_cov, w))
                p_vol = np.sqrt(max(p_var, 1e-8))
                # Marginal risk contribution
                mrc = np.dot(sub_cov, w) / p_vol
                # Risk contribution of each asset
                rc = w * mrc
                # Target risk contribution = p_vol / n_assets
                target_rc = p_vol / n_assets
                return np.sum(np.square(rc - target_rc))

        else:
            raise ValueError(f"Unknown optimization mode: {mode}")

        # Execute optimization
        res = minimize(
            objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'ftol': 1e-9, 'maxiter': 500}
        )

        final_weights = res.x if res.success else w0

        # Build output dictionary with rounding
        out_weights = {}
        for i, t in enumerate(tickers):
            out_weights[t] = round(float(final_weights[i]), 4)
        out_weights["Cash"] = round(target_cash, 4)

        # Normalize slight rounding differences so total == 1.0
        total_w = sum(out_weights.values())
        if abs(total_w - 1.0) > 1e-4:
            out_weights["Cash"] = round(1.0 - sum(v for k, v in out_weights.items() if k != "Cash"), 4)

        # Performance summary
        eq_weights = np.array([out_weights[t] for t in tickers])
        p_ret = float(np.sum(eq_weights * mu)) + target_cash * self.rf
        p_vol = float(np.sqrt(np.dot(eq_weights.T, np.dot(sub_cov, eq_weights))))
        p_vol = max(p_vol, 1e-6)
        p_sharpe = (p_ret - self.rf) / p_vol

        return {
            "mode": mode,
            "weights": out_weights,
            "metrics": {
                "expected_return": round(p_ret, 4),
                "expected_volatility": round(p_vol, 4),
                "sharpe": round(p_sharpe, 2)
            },
            "solver_success": bool(res.success)
        }


if __name__ == "__main__":
    from portfolio.risk_engine import PortfolioRiskEngine

    engine = PortfolioRiskEngine()
    cov = engine.get_covariance_matrix()

    returns = {
        "NVDA": 0.22,
        "AAPL": 0.12,
        "MSFT": 0.15,
        "AMZN": 0.14,
        "GOOGL": 0.13
    }
    sectors = {
        "NVDA": "Technology",
        "AAPL": "Technology",
        "MSFT": "Technology",
        "AMZN": "Consumer Cyclical",
        "GOOGL": "Communication"
    }

    opt = PortfolioOptimizer()
    print("--- 1. Max Sharpe Optimization (10% Cash) ---")
    sharpe_res = opt.optimize(returns, cov, mode="max_sharpe", cash_weight=0.10, sector_mapping=sectors)
    print(sharpe_res["weights"])
    print(sharpe_res["metrics"])

    print("\n--- 2. Minimum Variance Optimization ---")
    minvar_res = opt.optimize(returns, cov, mode="min_variance", cash_weight=0.15)
    print(minvar_res["weights"])
    print(minvar_res["metrics"])

    print("\n--- 3. Risk Parity Optimization ---")
    rp_res = opt.optimize(returns, cov, mode="risk_parity", cash_weight=0.10)
    print(rp_res["weights"])
    print(rp_res["metrics"])
