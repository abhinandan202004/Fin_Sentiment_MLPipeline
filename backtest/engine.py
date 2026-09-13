import logging
from typing import Dict, Any, List
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BacktestEngine")


class BacktestEngine:
    def __init__(self, risk_free_rate: float = 0.04):
        self.risk_free_rate = risk_free_rate

    def evaluate_strategy(
        self,
        test_df: pd.DataFrame,
        confidence_threshold: float = 0.50,
        holding_horizon: str = "return_5d",
    ) -> Dict[str, Any]:
        """
        Backtests strategy on test dataset.
        Enters a BUY position when model prediction confidence >= threshold.
        """
        df = test_df.copy().reset_index(drop=True)

        if holding_horizon not in df.columns:
            raise ValueError(f"Target return column {holding_horizon} not present in test data.")

        # Filter signals
        df["signal"] = (df["confidence"] >= confidence_threshold).astype(int)
        trades = df[df["signal"] == 1].copy()

        total_bars = len(df)
        total_trades = len(trades)

        if total_trades == 0:
            return {
                "total_trades": 0,
                "message": f"No trades triggered at confidence threshold {confidence_threshold:.2f}",
            }

        raw_returns = trades[holding_horizon].values
        # Filter out trailing NaNs if horizon extends beyond available future bars
        trade_returns = raw_returns[~np.isnan(raw_returns)]

        if len(trade_returns) == 0:
            return {
                "total_trades": 0,
                "message": f"No valid trades with realized returns for {holding_horizon}",
            }

        winning_trades = int((trade_returns > 0).sum())
        losing_trades = int((trade_returns < 0).sum())
        win_rate = winning_trades / len(trade_returns)

        avg_return = float(np.mean(trade_returns))
        cumulative_strategy_return = float(np.prod(1 + trade_returns) - 1)

        # Benchmark: Buy & Hold all periods in test set
        benchmark_returns = df[holding_horizon].dropna().values
        benchmark_cumulative = (
            float(np.prod(1 + benchmark_returns) - 1) if len(benchmark_returns) > 0 else 0.0
        )

        # Sharpe Ratio (annualized, adjusting for holding horizon)
        days_per_trade = 5 if "5d" in holding_horizon else (1 if "1d" in holding_horizon else 10)
        periods_per_year = 252 / days_per_trade
        std_return = float(np.std(trade_returns))
        rf_per_period = self.risk_free_rate / periods_per_year

        if std_return > 1e-6:
            sharpe_ratio = float((avg_return - rf_per_period) / std_return * np.sqrt(periods_per_year))
        else:
            sharpe_ratio = 0.0

        # Maximum Drawdown on cumulative equity
        equity_curve = np.cumprod(1 + trade_returns)
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - running_max) / running_max
        max_drawdown = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

        results = {
            "holding_horizon": holding_horizon,
            "confidence_threshold": confidence_threshold,
            "total_test_days": total_bars,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": round(win_rate * 100, 2),
            "average_return_per_trade": round(avg_return * 100, 2),
            "cumulative_strategy_return": round(cumulative_strategy_return * 100, 2),
            "benchmark_cumulative_return": round(benchmark_cumulative * 100, 2),
            "sharpe_ratio": round(sharpe_ratio, 2),
            "max_drawdown": round(max_drawdown * 100, 2),
        }

        logger.info(
            f"Backtest [{holding_horizon} | Thresh={confidence_threshold:.2f}]: "
            f"Trades={total_trades}, WinRate={results['win_rate']}%, "
            f"CumReturn={results['cumulative_strategy_return']}%, "
            f"Sharpe={results['sharpe_ratio']}, MaxDD={results['max_drawdown']}%"
        )
        return results


def run_backtest(
    test_df: pd.DataFrame,
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """Runs backtest across 1-day, 5-day, and 10-day horizons."""
    engine = BacktestEngine()
    results = {}
    for horizon in ["return_1d", "return_5d", "return_10d"]:
        if horizon in test_df.columns:
            results[horizon] = engine.evaluate_strategy(
                test_df, confidence_threshold=threshold, holding_horizon=horizon
            )
    return results
