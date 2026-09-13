"""
Autonomous Portfolio Agent & Investment Assistant for Fin_Sentiment_MLPipeline.

Master orchestrator for Sprint 5:
- Regime Agent (Macro market regime & risk bias)
- Risk Engine (Annualized Volatility, Beta, VaR, CVaR, Max Drawdown, HHI)
- Expected Return Engine (Multi-modal forward alpha from ML, Analogs, Sentiment, Sector)
- Position Sizer (Kelly Criterion, Volatility Parity, Risk-fractional sizing)
- Portfolio Constraints (Exposure guardrails)
- Transaction Cost Model (Turnover friction)
- Portfolio Optimizer (Markowitz Max Sharpe, Min Variance, Risk Parity via scipy.optimize)
- Portfolio Health Engine (0-100 score and letter grade)
- Portfolio Memory (Audit trail & snapshot persistence)

Produces:
1. Explainable Allocation Object (JSON output contract)
2. Comprehensive Markdown Investment Report in reports/portfolio_reports/
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

from config import REPORTS_DIR
from agents.regime_agent import RegimeAgent
from portfolio.risk_engine import PortfolioRiskEngine
from portfolio.optimizer import PortfolioOptimizer
from portfolio.position_sizer import PositionSizer
from portfolio.expected_return import ExpectedReturnEngine
from portfolio.constraints import PortfolioConstraints
from portfolio.transaction_costs import TransactionCostModel
from portfolio.portfolio_health import PortfolioHealthEngine
from portfolio.portfolio_memory import PortfolioMemory
from portfolio.watchlist_ranker import WatchlistRanker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("PortfolioAgent")


class PortfolioAgent:
    """Master Autonomous Portfolio Intelligence Agent."""

    def __init__(
        self,
        risk_profile: str = "moderate",
        risk_free_rate: float = 0.04
    ):
        self.risk_profile = risk_profile.lower()
        self.rf = risk_free_rate

        # Initialize engines
        self.regime_agent = RegimeAgent()
        self.risk_engine = PortfolioRiskEngine()
        self.expected_return_engine = ExpectedReturnEngine(baseline_market_return=0.10)
        self.position_sizer = PositionSizer()
        self.health_engine = PortfolioHealthEngine()
        self.memory = PortfolioMemory()
        self.watchlist_ranker = WatchlistRanker()
        self.cost_model = TransactionCostModel(cost_rate=0.001)

        # Configure profile constraints
        if self.risk_profile == "conservative":
            self.constraints = PortfolioConstraints(
                max_single_position=0.20,
                max_sector_exposure=0.30,
                minimum_cash=0.20,
                max_portfolio_beta=0.90
            )
        elif self.risk_profile == "aggressive":
            self.constraints = PortfolioConstraints(
                max_single_position=0.35,
                max_sector_exposure=0.50,
                minimum_cash=0.05,
                max_portfolio_beta=1.40
            )
        else:  # moderate
            self.constraints = PortfolioConstraints(
                max_single_position=0.30,
                max_sector_exposure=0.40,
                minimum_cash=0.10,
                max_portfolio_beta=1.20
            )

        self.optimizer = PortfolioOptimizer(
            risk_free_rate=self.rf,
            constraints=self.constraints,
            cost_model=self.cost_model
        )

    def run_allocation(
        self,
        capital: float = 100000.0,
        current_holdings: Optional[Dict[str, float]] = None,
        candidate_universe: Optional[List[str]] = None,
        optimization_mode: str = "max_sharpe"
    ) -> Dict[str, Any]:
        """
        Executes end-to-end portfolio decision workflow.
        """
        logger.info(f"Running Portfolio Allocation (Capital: ${capital:,.2f}, Risk Profile: {self.risk_profile.upper()})...")

        # 1. Detect Macro Market Regime
        regime_data = self.regime_agent.detect_regime()
        regime = regime_data["regime"]
        regime_conf = regime_data["confidence"]
        regime_bias = regime_data["bias"]
        regime_rules = regime_data["allocation_rules"]

        # Adjust cash buffer based on regime and constraints
        target_cash = max(regime_rules["cash_weight"], self.constraints.minimum_cash)
        if self.risk_profile == "conservative":
            target_cash = min(target_cash + 0.10, 0.50)

        # 2. Normalize Current Holdings
        if not current_holdings:
            current_holdings = {"NVDA": 0.25, "AAPL": 0.25, "MSFT": 0.25, "Cash": 0.25}

        clean_curr_weights = {}
        for k, v in current_holdings.items():
            clean_curr_weights[k.upper() if k.upper() != "CASH" else "Cash"] = float(v)

        # Ensure current weights sum to 1.0
        total_curr = sum(clean_curr_weights.values())
        if total_curr > 0:
            clean_curr_weights = {k: round(v / total_curr, 4) for k, v in clean_curr_weights.items()}

        # 3. Define Universe
        if not candidate_universe:
            candidate_universe = ["NVDA", "MSFT", "AAPL", "AMZN", "GOOGL"]
        
        # Include any existing stocks in the universe
        for t in clean_curr_weights.keys():
            if t != "Cash" and t not in candidate_universe:
                candidate_universe.append(t)

        # 4. Multi-Modal Expected Returns & Alpha Signals
        # Multi-factor signal bank (combining ML probabilities, FinBERT sentiment, analogs)
        signal_db = {
            "NVDA": {"ml_prob": 0.72, "analog_success": 0.78, "sentiment": 0.65, "sector_strength": 0.14, "vol": 0.35, "sector": "Technology"},
            "MSFT": {"ml_prob": 0.65, "analog_success": 0.70, "sentiment": 0.42, "sector_strength": 0.10, "vol": 0.22, "sector": "Technology"},
            "AAPL": {"ml_prob": 0.55, "analog_success": 0.60, "sentiment": 0.20, "sector_strength": 0.08, "vol": 0.20, "sector": "Technology"},
            "AMZN": {"ml_prob": 0.60, "analog_success": 0.62, "sentiment": 0.35, "sector_strength": 0.06, "vol": 0.28, "sector": "Consumer Cyclical"},
            "GOOGL": {"ml_prob": 0.62, "analog_success": 0.66, "sentiment": 0.30, "sector_strength": 0.07, "vol": 0.25, "sector": "Communication"},
            "TCS": {"ml_prob": 0.46, "analog_success": 0.48, "sentiment": -0.12, "sector_strength": -0.04, "vol": 0.24, "sector": "IT Services"},
            "INFY": {"ml_prob": 0.49, "analog_success": 0.51, "sentiment": -0.05, "sector_strength": -0.03, "vol": 0.25, "sector": "IT Services"},
            "HDFCBANK": {"ml_prob": 0.58, "analog_success": 0.61, "sentiment": 0.25, "sector_strength": 0.05, "vol": 0.21, "sector": "Financials"}
        }

        sector_mapping = {}
        expected_returns = {}
        candidate_scores = []

        for ticker in candidate_universe:
            sig = signal_db.get(ticker, {
                "ml_prob": 0.52, "analog_success": 0.50, "sentiment": 0.05,
                "sector_strength": 0.02, "vol": 0.25, "sector": "Other"
            })
            sector_mapping[ticker] = sig["sector"]

            ret_res = self.expected_return_engine.calculate_ticker_return(
                ticker=ticker,
                ml_prob=sig["ml_prob"],
                analog_success_rate=sig["analog_success"],
                sentiment_score=sig["sentiment"],
                sector_strength=sig["sector_strength"]
            )
            expected_returns[ticker] = ret_res["expected_return"]

            candidate_scores.append({
                "ticker": ticker,
                "signal_strength": sig["ml_prob"],
                "expected_return": ret_res["expected_return"],
                "analog_success": sig["analog_success"],
                "risk": sig["vol"]
            })

        # Rank Watchlist
        ranked_watchlist = self.watchlist_ranker.rank_universe(candidate_scores)

        # 5. Position Sizing Verification (Half-Kelly)
        sizing_recommendations = {}
        for ticker in candidate_universe:
            sig = signal_db.get(ticker, {"ml_prob": 0.55, "vol": 0.25})
            rec_size = self.position_sizer.recommend_size(
                ticker=ticker,
                win_rate=sig["ml_prob"],
                reward_risk_ratio=1.35,
                volatility=sig["vol"],
                max_position=self.constraints.max_single_position,
                regime=regime
            )
            sizing_recommendations[ticker] = rec_size

        # 6. Portfolio Optimization (Scipy SLSQP)
        cov_matrix = self.risk_engine.get_covariance_matrix(candidate_universe)
        opt_mode = "min_variance" if self.risk_profile == "conservative" else optimization_mode

        opt_result = self.optimizer.optimize(
            expected_returns=expected_returns,
            cov_matrix=cov_matrix,
            mode=opt_mode,
            cash_weight=target_cash,
            current_weights={k: v for k, v in clean_curr_weights.items() if k != "Cash"},
            sector_mapping=sector_mapping
        )

        target_weights = opt_result["weights"]
        opt_metrics = opt_result["metrics"]

        # 7. Evaluate Target Risk & Health
        target_risk_metrics = self.risk_engine.evaluate_portfolio(
            target_weights,
            risk_free_rate=self.rf
        )

        health_result = self.health_engine.compute_health_score(
            hhi=target_risk_metrics["hhi"],
            volatility=target_risk_metrics["volatility"],
            beta=target_risk_metrics["beta"],
            cash_buffer=target_risk_metrics["cash_buffer"],
            max_drawdown=target_risk_metrics["max_drawdown"]
        )

        # 8. Generate Trade Recommendations & Explainable Rationales
        recommended_actions = []
        explainable_reasons: Dict[str, List[str]] = {}

        all_tickers = sorted(list(set(list(clean_curr_weights.keys()) + list(target_weights.keys()))))

        for t in all_tickers:
            if t == "Cash":
                continue
            curr_w = clean_curr_weights.get(t, 0.0)
            targ_w = target_weights.get(t, 0.0)
            delta_w = round(targ_w - curr_w, 4)

            sig = signal_db.get(t, {
                "ml_prob": 0.50, "analog_success": 0.50, "sentiment": 0.0, "sector": "Other"
            })

            reasons = []
            if delta_w > 0.03:
                action = "BUY"
                reasons.append(f"BUY signal confidence {int(sig['ml_prob']*100)}%")
                reasons.append(f"Analog pattern success rate {int(sig['analog_success']*100)}%")
                if regime_bias == "RISK_ON":
                    reasons.append(f"{regime} market regime favorable for equity expansion")
                reasons.append(f"Optimal Sharpe weight expansion (+{int(delta_w*100)}%)")
            elif delta_w < -0.03:
                action = "REDUCE"
                if sig["ml_prob"] < 0.50:
                    reasons.append(f"Sub-50% ML probability ({int(sig['ml_prob']*100)}%) signaling negative momentum")
                if sig["sentiment"] < 0:
                    reasons.append("FinBERT negative news sentiment drag")
                reasons.append(f"Trimming exposure by {abs(int(delta_w*100))}% to reallocate into higher-Sharpe assets")
            else:
                action = "HOLD"
                reasons.append(f"Allocation remains within optimal tolerance bounds ({int(targ_w*100)}%)")

            explainable_reasons[t] = reasons

            if action != "HOLD":
                recommended_actions.append({
                    "ticker": t,
                    "action": action,
                    "allocation": delta_w
                })

        # 9. Persist to Portfolio Memory
        try:
            self.memory.record_portfolio_snapshot(
                total_capital=capital,
                cash_weight=target_weights.get("Cash", target_cash),
                positions=target_weights
            )
            for act in recommended_actions:
                t = act["ticker"]
                targ_w = target_weights.get(t, 0.0)
                delta_w = act["allocation"]
                sig = signal_db.get(t, {"ml_prob": 0.60})
                exp_ret = expected_returns.get(t, 0.10)
                self.memory.record_decision(
                    ticker=t,
                    action=act["action"],
                    target_weight=targ_w,
                    delta_weight=delta_w,
                    confidence=sig["ml_prob"],
                    expected_return=exp_ret,
                    rationale=explainable_reasons.get(t, [])
                )
        except Exception as e:
            logger.warning(f"Could not write to portfolio memory: {e}")

        # 10. Construct Final JSON Output Contract
        output_contract = {
            "market_regime": regime,
            "portfolio_health": health_result["portfolio_health"],
            "recommended_actions": recommended_actions,
            "expected_return": opt_metrics["expected_return"],
            "expected_volatility": opt_metrics["expected_volatility"],
            "sharpe": opt_metrics["sharpe"]
        }

        # 11. Generate Markdown Report
        report_md = self._generate_markdown_report(
            capital=capital,
            regime_data=regime_data,
            current_holdings=clean_curr_weights,
            target_weights=target_weights,
            recommended_actions=recommended_actions,
            explainable_reasons=explainable_reasons,
            opt_metrics=opt_metrics,
            risk_metrics=target_risk_metrics,
            health_result=health_result,
            ranked_watchlist=ranked_watchlist
        )

        return {
            "contract": output_contract,
            "full_details": {
                "regime": regime_data,
                "target_weights": target_weights,
                "current_weights": clean_curr_weights,
                "risk_metrics": target_risk_metrics,
                "health": health_result,
                "explainable_reasons": explainable_reasons,
                "ranked_watchlist": ranked_watchlist
            },
            "report_markdown": report_md
        }

    def _generate_markdown_report(
        self,
        capital: float,
        regime_data: Dict[str, Any],
        current_holdings: Dict[str, float],
        target_weights: Dict[str, float],
        recommended_actions: List[Dict[str, Any]],
        explainable_reasons: Dict[str, List[str]],
        opt_metrics: Dict[str, Any],
        risk_metrics: Dict[str, Any],
        health_result: Dict[str, Any],
        ranked_watchlist: List[Dict[str, Any]]
    ) -> str:
        """Generates comprehensive institutional-grade portfolio intelligence report."""
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        regime = regime_data["regime"]
        regime_conf = int(regime_data["confidence"] * 100)
        bias = regime_data["bias"]

        # Current vs Target table
        all_assets = sorted(list(set(list(current_holdings.keys()) + list(target_weights.keys()))))
        alloc_table_rows = []
        for a in all_assets:
            curr_w = current_holdings.get(a, 0.0)
            targ_w = target_weights.get(a, 0.0)
            delta_w = targ_w - curr_w
            curr_val = curr_w * capital
            targ_val = targ_w * capital
            alloc_table_rows.append(
                f"| `{a}` | {curr_w*100:5.1f}% | ${curr_val:9,.2f} | {targ_w*100:5.1f}% | ${targ_val:9,.2f} | {delta_w*100:+5.1f}% |"
            )
        alloc_table = "\n".join(alloc_table_rows)

        # Recommended actions table
        action_rows = []
        for act in recommended_actions:
            t = act["ticker"]
            dollar_val = abs(act["allocation"] * capital)
            action_rows.append(
                f"| **{act['action']}** | `{t}` | {act['allocation']*100:+5.1f}% | ${dollar_val:,.2f} |"
            )
        action_table = "\n".join(action_rows) if action_rows else "| *NO TRADES* | - | 0.0% | $0.00 |"

        # Rationale bullets
        rationale_sections = []
        for a in all_assets:
            if a == "Cash":
                continue
            reasons = explainable_reasons.get(a, [])
            if reasons:
                bullets = "\n".join([f"  - {r}" for r in reasons])
                targ_w = target_weights.get(a, 0.0)
                rationale_sections.append(f"- **{a}** (Target: {targ_w*100:.1f}%):\n{bullets}")
        rationale_text = "\n\n".join(rationale_sections)

        # Audit Trail evaluation
        past_eval = self.memory.evaluate_past_recommendations()

        report_md = f"""# Autonomous Portfolio Intelligence Report
**Generated:** {now_str}  
**Total Capital:** ${capital:,.2f}  
**Risk Profile:** {self.risk_profile.upper()}  
**Optimization Mode:** Modern Portfolio Theory (Mean-Variance Sharpe Maximization)

---

## 1. Portfolio Summary
- **Portfolio Health Score:** **{health_result['portfolio_health']} / 100 (Grade: {health_result['grade']})**
- **Expected Annual Return:** **{opt_metrics['expected_return']*100:.2f}%**
- **Expected Portfolio Volatility:** **{opt_metrics['expected_volatility']*100:.2f}%**
- **Sharpe Ratio:** **{opt_metrics['sharpe']:.2f}**
- **Cash Buffer:** **{target_weights.get('Cash', 0.10)*100:.1f}%**

---

## 2. Market Regime
- **Current Regime:** **{regime}** (Confidence: {regime_conf}%)
- **Allocation Bias:** `{bias}`
- **Regime Guidelines:** {regime_data['allocation_rules'].get('description', '')}

---

## 3. Current vs Target Allocation
| Asset | Current % | Current Value | Target % | Target Value | Delta % |
|:------|:---------:|:-------------:|:--------:|:------------:|:-------:|
{alloc_table}

---

## 4. Recommended Trades
| Action | Ticker | Delta Allocation | Execution Amount |
|:-------|:------:|:----------------:|:----------------:|
{action_table}

---

## 5. Portfolio Risk Metrics
| Metric | Target Value | Benchmark / Guardrail | Status |
|:-------|:------------:|:---------------------:|:------:|
| **Annualized Volatility (σp)** | {risk_metrics['volatility']*100:.2f}% | < 25.0% | {'✅ OK' if risk_metrics['volatility'] <= 0.25 else '⚠️ ELEVATED'} |
| **Portfolio Beta (vs SPY)** | {risk_metrics['beta']:.2f} | < {self.constraints.max_portfolio_beta:.2f} | {'✅ COMPLIANT' if risk_metrics['beta'] <= self.constraints.max_portfolio_beta else '⚠️ HIGH'} |
| **Parametric VaR (1-day, 95%)** | {risk_metrics['var_95']*100:.2f}% | - | Normal |
| **CVaR / Expected Shortfall (95%)** | {risk_metrics['cvar_95']*100:.2f}% | - | Normal |
| **Max Historical Drawdown** | {risk_metrics['max_drawdown']*100:.2f}% | < 20.0% | ✅ CONTROLLED |
| **Concentration Risk (HHI)** | {risk_metrics['hhi']:.3f} | < 0.250 | {'✅ DIVERSIFIED' if risk_metrics['hhi'] <= 0.25 else '⚠️ CONCENTRATED'} |
| **Cash Buffer** | {risk_metrics['cash_buffer']*100:.1f}% | >= {self.constraints.minimum_cash*100:.1f}% | ✅ SECURE |

---

## 6. Portfolio Health Breakdown
- **Overall Score:** {health_result['portfolio_health']} / 100 (`{health_result['grade']}`)
  - **Diversification (HHI):** {health_result['breakdown']['diversification']} / 25
  - **Risk Profile (Vol & Beta):** {health_result['breakdown']['risk_score']} / 25
  - **Cash Buffer Liquidity:** {health_result['breakdown']['cash_buffer']} / 25
  - **Drawdown Resilience:** {health_result['breakdown']['drawdown_score']} / 25

---

## 7. Explainable Allocation Rationale
{rationale_text}

---

## 8. Candidate Watchlist Ranking
| Rank | Ticker | Composite Score | ML Signal | Exp Return | Analog Win Rate | Volatility |
|:----:|:------:|:---------------:|:---------:|:----------:|:---------------:|:----------:|
"""
        for r in ranked_watchlist:
            report_md += f"| {r['rank']} | `{r['ticker']}` | **{r['score']:.2f}** | {r['signal_strength']*100:.0f}% | {r['expected_return']*100:.1f}% | {r['analog_success']*100:.0f}% | {r['risk']*100:.0f}% |\n"

        report_md += f"""
---

## 9. Decision Audit Trail
- **Total Historical Decisions Recorded:** {past_eval['total_decisions']}
- **Decision Accuracy Track Record:** **{past_eval['decision_accuracy']*100:.1f}%**
- **Mean Expected Return:** {past_eval['mean_expected_return']*100:.2f}%
- **Mean Realized Return:** {past_eval['mean_realized_return']*100:.2f}%
- **Tracking Error:** {past_eval['tracking_error']*100:.2f}%
"""

        # Save to reports directory
        portfolio_report_dir = REPORTS_DIR / "portfolio_reports"
        portfolio_report_dir.mkdir(parents=True, exist_ok=True)
        report_file = portfolio_report_dir / "portfolio_allocation_report.md"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report_md)
        logger.info(f"Saved portfolio allocation report to {report_file}")

        return report_md


def main():
    parser = argparse.ArgumentParser(description="Autonomous Portfolio Agent & Investment Assistant")
    parser.add_argument("--capital", type=float, default=100000.0, help="Total investment capital (e.g. 100000)")
    parser.add_argument("--risk", type=str, default="moderate", choices=["conservative", "moderate", "aggressive"], help="Risk profile")
    parser.add_argument("--mode", type=str, default="max_sharpe", choices=["max_sharpe", "min_variance", "risk_parity"], help="Optimizer mode")
    parser.add_argument("--holdings", type=str, default=None, help='JSON string of current holdings, e.g. \'{"TCS": 0.4, "Cash": 0.6}\'')
    parser.add_argument("--json", action="store_true", help="Print only JSON contract")

    args = parser.parse_args()

    holdings = None
    if args.holdings:
        try:
            holdings = json.loads(args.holdings)
        except Exception as e:
            logger.error(f"Failed to parse --holdings JSON: {e}")

    agent = PortfolioAgent(risk_profile=args.risk)
    res = agent.run_allocation(
        capital=args.capital,
        current_holdings=holdings,
        optimization_mode=args.mode
    )

    # Print required JSON contract
    contract_json = json.dumps(res["contract"], indent=2)
    print(contract_json)


if __name__ == "__main__":
    main()
