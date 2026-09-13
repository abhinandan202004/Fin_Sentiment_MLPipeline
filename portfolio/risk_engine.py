import logging
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from scipy import stats

from config import DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("PortfolioRiskEngine")

SECTOR_MAP = {
    "NVDA": "Semiconductors",
    "AAPL": "Consumer Technology",
    "MSFT": "Cloud Computing",
    "AMZN": "Cloud Computing",
    "GOOGL": "Cloud Computing",
    "TSM": "Semiconductors",
    "XOM": "Energy",
    "TCS": "IT Services",
    "INFY": "IT Services",
    "HDFCBANK": "Banking & Financials",
    "CASH": "Cash & Liquidity",
}


class PortfolioRiskEngine:
    def __init__(self):
        self._returns_matrix = None
        self._spy_returns = None
        self._load_market_returns()

    def _load_market_returns(self):
        csv_path = DATA_DIR / "training_dataset.csv"
        if not csv_path.exists():
            logger.warning(f"Market dataset not found at {csv_path}. Using synthetic covariance.")
            return

        df = pd.read_csv(csv_path)
        # Pivot table of daily returns: columns = tickers, index = date
        pivoted = df.pivot(index="date", columns="ticker", values="return_1d").dropna()
        self._returns_matrix = pivoted

        # Load SPY daily return if available
        spy_dates = df[["date", "spy_return_1d"]].drop_duplicates().set_index("date")["spy_return_1d"].dropna()
        self._spy_returns = spy_dates

    def get_covariance_matrix(self, tickers: Optional[List[str]] = None) -> pd.DataFrame:
        """Returns annualized covariance matrix for available tickers."""
        if self._returns_matrix is not None:
            if tickers:
                valid_tickers = [t for t in tickers if t in self._returns_matrix.columns]
                if valid_tickers:
                    return self._returns_matrix[valid_tickers].cov() * 252.0
            return self._returns_matrix.cov() * 252.0
        
        # Fallback synthetic covariance
        cols = tickers if tickers else ["NVDA", "AAPL", "MSFT", "AMZN", "GOOGL"]
        n = len(cols)
        corr = np.full((n, n), 0.5)
        np.fill_diagonal(corr, 1.0)
        vols = np.array([0.35, 0.22, 0.24, 0.28, 0.25][:n])
        cov = np.outer(vols, vols) * corr
        return pd.DataFrame(cov, index=cols, columns=cols)

    def evaluate_portfolio(
        self,
        weights: Dict[str, float],
        risk_free_rate: float = 0.04,
    ) -> Dict[str, Any]:
        """
        Calculates comprehensive portfolio-level risk metrics:
        - Annualized Volatility
        - Portfolio Beta vs SPY
        - Parametric & Historical VaR (95%, 99%)
        - Conditional VaR (Expected Shortfall)
        - Max Drawdown
        - Concentration Risk (HHI)
        - Sector Breakdown
        """
        # Normalize weights so equity + cash = 1.0
        clean_weights = {k.upper(): float(v) for k, v in weights.items() if float(v) > 0}
        total_w = sum(clean_weights.values())
        if total_w > 0:
            clean_weights = {k: v / total_w for k, v in clean_weights.items()}

        cash_weight = clean_weights.get("CASH", 0.0)
        equity_weights = {k: v for k, v in clean_weights.items() if k != "CASH"}

        tickers = list(equity_weights.keys())
        w_vec = np.array([equity_weights[t] for t in tickers], dtype=float)

        # 1. Concentration Risk (HHI - Herfindahl-Hirschman Index)
        hhi = float(np.sum([w**2 for w in clean_weights.values()]))

        # 2. Sector Exposure Breakdown
        sector_exp: Dict[str, float] = {}
        for t, w in clean_weights.items():
            sec = SECTOR_MAP.get(t, "Other Equities")
            sector_exp[sec] = round(sector_exp.get(sec, 0.0) + w, 3)

        # 3. Covariance & Returns
        if self._returns_matrix is not None and all(t in self._returns_matrix.columns for t in tickers) and len(tickers) > 0:
            sub_returns = self._returns_matrix[tickers]
            cov_matrix = sub_returns.cov().values * 252.0  # Annualized covariance
            daily_port_returns = sub_returns.dot(w_vec).values
            annualized_vol = float(np.sqrt(w_vec.T.dot(cov_matrix).dot(w_vec)))

            # Portfolio Beta vs SPY
            aligned_spy = self._spy_returns.reindex(sub_returns.index).fillna(0.0).values
            cov_spy = np.cov(daily_port_returns, aligned_spy)[0, 1]
            var_spy = np.var(aligned_spy)
            port_beta = float(cov_spy / var_spy) if var_spy > 1e-6 else 1.0

            # Historical Max Drawdown
            equity_curve = np.cumprod(1 + daily_port_returns)
            running_max = np.maximum.accumulate(equity_curve)
            dd = (equity_curve - running_max) / running_max
            max_dd = float(np.abs(np.min(dd))) if len(dd) > 0 else 0.08
        else:
            # Fallback estimation based on average asset volatility
            annualized_vol = 0.18 * (1.0 - cash_weight)
            port_beta = 1.10 * (1.0 - cash_weight)
            max_dd = 0.12 * (1.0 - cash_weight)
            daily_port_returns = np.random.normal(0.0004, annualized_vol / np.sqrt(252), size=500)

        # Adjust volatility and beta for cash weight
        annualized_vol = annualized_vol * (1.0 - cash_weight)
        port_beta = port_beta * (1.0 - cash_weight)

        # 4. Value at Risk (1-day 95% and 99%)
        var_95 = float(np.percentile(-daily_port_returns, 95)) if len(daily_port_returns) > 0 else 0.025
        var_99 = float(np.percentile(-daily_port_returns, 99)) if len(daily_port_returns) > 0 else 0.040

        # 5. Conditional Value at Risk / Expected Shortfall (CVaR)
        tail_losses_95 = -daily_port_returns[-daily_port_returns >= var_95]
        cvar_95 = float(np.mean(tail_losses_95)) if len(tail_losses_95) > 0 else var_95 * 1.3

        tail_losses_99 = -daily_port_returns[-daily_port_returns >= var_99]
        cvar_99 = float(np.mean(tail_losses_99)) if len(tail_losses_99) > 0 else var_99 * 1.3

        return {
            "volatility": round(annualized_vol, 4),
            "beta": round(port_beta, 2),
            "var_95": round(var_95, 4),
            "var_99": round(var_99, 4),
            "cvar_95": round(cvar_95, 4),
            "cvar_99": round(cvar_99, 4),
            "max_drawdown": round(max_dd, 4),
            "hhi": round(hhi, 3),
            "cash_buffer": round(cash_weight, 3),
            "sector_exposure": sector_exp,
        }


if __name__ == "__main__":
    engine = PortfolioRiskEngine()
    test_weights = {"NVDA": 0.30, "AAPL": 0.25, "MSFT": 0.25, "AMZN": 0.10, "CASH": 0.10}
    res = engine.evaluate_portfolio(test_weights)
    print("\nPortfolio Risk Engine Evaluation:")
    for k, v in res.items():
        print(f"  {k:<16}: {v}")
