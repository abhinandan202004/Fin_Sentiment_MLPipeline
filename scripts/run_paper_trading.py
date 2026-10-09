"""
CLI Script for Phase 8: Walk-Forward Paper Trading.
Sprint 9: Autonomous Evaluation & Adaptive Intelligence.

Executes walk-forward portfolio simulation with risk controls and transaction costs:
Usage:
    python scripts/run_paper_trading.py --initial-capital 1000000
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from evaluation.paper_trader import PaperTrader

def main():
    parser = argparse.ArgumentParser(description="Sprint 9 Walk-Forward Paper Trading Simulator")
    parser.add_argument("--initial-capital", type=float, default=1_000_000.0, help="Initial portfolio capital in INR (default: 10,00,000)")
    args = parser.parse_args()

    print("\n=======================================================")
    print(" SPRINT 9: WALK-FORWARD PAPER TRADING SIMULATION")
    print(f" Initial Capital: INR {args.initial_capital:,.2f}")
    print("=======================================================\n")

    trader = PaperTrader(initial_capital=args.initial_capital)
    results = trader.run_walk_forward(persist_to_db=True)
    report_file = trader.generate_trading_report(results)

    print(f"Final Portfolio Equity:  INR {results['final_equity']:,.2f}")
    print(f"Total Cumulative Return: {results['cumulative_return']:+.2%}")
    print(f"Benchmark (SPY) Return:  {results['benchmark_return']:+.2%}")
    print(f"Cumulative Alpha:        {results['alpha']:+.2%}")
    print(f"Sharpe Ratio:            {results['sharpe_ratio']:.2f}")
    print(f"Sortino Ratio:           {results['sortino_ratio']:.2f}")
    print(f"Maximum Drawdown:        {results['max_drawdown']:.2%}")
    print(f"Profit Factor:           {results['profit_factor']:.2f}")
    print(f"Trades Executed:         {results['total_closed_trades']} closed ({results['hit_rate']:.1%} win rate)")
    print(f"Total Friction Deducted: INR {results['total_transaction_costs']:,.2f}")
    print(f"\nReport Generated:        {report_file}")
    print("=======================================================\n")

if __name__ == "__main__":
    main()
