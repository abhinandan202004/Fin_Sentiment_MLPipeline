import logging
import sys
import json
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict, Any, List
from datetime import datetime, timezone

from config import BASE_DIR
from agents.evidence import Evidence
from agents.news_agent import NewsAgent
from agents.event_agent import EventAgent
from agents.technical_agent import TechnicalAgent
from agents.analog_agent import AnalogAgent
from agents.risk_agent import RiskAgent
from agents.analyst_agent import AnalystAgent
from graph.graph_queries import GraphQueryEngine
from models.predict import PredictionService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ReportGenerator")

REPORTS_DIR = BASE_DIR / "reports" / "research_reports"


class ResearchReportGenerator:
    def __init__(self):
        self.news_agent = NewsAgent()
        self.event_agent = EventAgent()
        self.technical_agent = TechnicalAgent()
        self.analog_agent = AnalogAgent()
        self.risk_agent = RiskAgent()
        self.analyst_agent = AnalystAgent()
        self.graph_engine = GraphQueryEngine.get_instance()
        self.prediction_service = PredictionService()

    def generate_report(self, ticker: str, save_markdown: bool = True) -> Dict[str, Any]:
        ticker = ticker.upper().strip()
        logger.info(f"Generating full AI Financial Research Report for {ticker}...")

        # 1. ML Prediction & SHAP Drivers
        ml_pred = self.prediction_service.predict(ticker)
        signal = ml_pred.get("signal", ml_pred.get("prediction", "NOT BUY"))
        probability = float(ml_pred.get("probability", 0.50))
        threshold = float(ml_pred.get("threshold", 0.55))
        confidence = float(ml_pred.get("confidence", 0.50))
        top_pos = ml_pred.get("top_positive_drivers", [])
        top_neg = ml_pred.get("top_negative_drivers", [])
        top_drivers = top_pos + top_neg
        quant_features = ml_pred.get("quant_features", {})
        market_regime = ml_pred.get("market_regime", "Neutral")
        feature_contributions = ml_pred.get("feature_contributions", [])

        # 2. Event Detection Engine
        raw_events = self.event_agent.detect_events_for_ticker(ticker, limit=6)
        event_names = [e["event"] for e in raw_events] if raw_events else ["routine_disclosure"]

        # 3. Financial Knowledge Graph
        graph_impacts = self.graph_engine.get_ticker_graph_impacts(ticker)
        if not graph_impacts:
            graph_impacts = [f"Direct equity monitoring for {ticker}"]

        # 4. Historical Analog Search (FAISS 30k Engine)
        analog_results = self.analog_agent.analyze_analogs(ticker, top_k=50)

        # 5. Technical & Risk Assessment
        tech_summary = self.technical_agent.analyze_ticker(ticker)
        risk_summary = self.risk_agent.evaluate_risk(ticker)

        # 6. Build Shared Evidence Contract (Sprint 4.1 Upgrade)
        evidence = Evidence(
            ticker=ticker,
            prediction=signal,
            confidence=confidence,
            raw_probability=probability,
            optimal_threshold=threshold,
            market_regime=market_regime,
            quant_features=quant_features,
            events=raw_events,
            graph_impacts=graph_impacts,
            analogs=analog_results,
            technical_summary=tech_summary,
            risk_summary=risk_summary,
            shap_drivers=top_drivers,
            feature_contributions=feature_contributions,
        )

        # 7. Analyst Agent Synthesis
        conclusion = self.analyst_agent.synthesize(evidence)

        # 8. API Contract Response
        response_payload = {
            "ticker": ticker,
            "signal": signal,
            "probability": round(probability, 4),
            "threshold": round(threshold, 2),
            "confidence": round(confidence, 4),
            "market_regime": market_regime,
            "events": event_names[:3],
            "graph_impacts": [graph_impacts[0]] if graph_impacts else ["Sector correlation"],
            "historical_analogs": {
                "sample_size": analog_results.get("sample_size", 50),
                "similar_cases": analog_results.get("similar_cases", 50),
                "success_rate": round(float(analog_results.get("success_rate", 0.50)), 4),
                "median_return_5d": round(float(analog_results.get("median_return_5d", 0.0)), 4),
                "confidence_interval": analog_results.get("confidence_interval", [0.0, 0.0]),
            },
            "top_shap_drivers": {
                "positive": top_pos[:3],
                "negative": top_neg[:3],
            },
            "quant_features": quant_features,
            "risks": risk_summary.get("risk_factors", ["Macro volatility"])[:2],
            "analyst_conclusion": conclusion,
        }

        # 9. Format and Save Markdown Report
        if save_markdown:
            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            report_file = REPORTS_DIR / f"{ticker}_research_report.md"
            self._save_markdown_report(ticker, evidence, response_payload, report_file)
            logger.info(f"Markdown research report written to {report_file}")

        return response_payload

    def _save_markdown_report(
        self,
        ticker: str,
        evidence: Evidence,
        payload: Dict[str, Any],
        report_file: Path,
    ):
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        events_md = ""
        for ev in evidence.events[:4]:
            events_md += f"- **{ev['event'].replace('_', ' ').title()}** ({ev['impact'].upper()}, score: {ev['impact_score']:+.2f}): *{ev['headline']}*\n"
        if not events_md:
            events_md = "- Routine corporate disclosures.\n"

        graph_md = ""
        for gi in evidence.graph_impacts[:4]:
            graph_md += f"- {gi}\n"

        analogs = evidence.analogs
        top_matches_md = ""
        for m in analogs.get("top_matches", [])[:4]:
            top_matches_md += f"- `{m['date']}` ({m['ticker']}, {m.get('sector', 'Other')}): 5d Return: `{m['return_5d']:+.2%}` | Regime: *{m.get('market_regime', 'Neutral')}* | Event: *{m['event']}* (Dist: {m['distance']})\n"

        risks_md = ""
        for r in evidence.risk_summary.get("risk_factors", []):
            risks_md += f"- {r}\n"

        pos_drivers_md = "\n".join([f"- **Positive**: `{d}`" for d in payload.get("top_shap_drivers", {}).get("positive", [])])
        neg_drivers_md = "\n".join([f"- **Headwind**: `{d}`" for d in payload.get("top_shap_drivers", {}).get("negative", [])])

        ci = analogs.get("confidence_interval", [0.0, 0.0])
        quant = evidence.quant_features
        tech = evidence.technical_summary

        content = f"""# Equity Research Note: {ticker}

**Generated**: {now_str}  
**Primary Recommendation**: **{payload['signal']}**  
**Calibrated Probability**: **{payload['probability']:.1%}** (Decision Threshold: `{payload['threshold']:.1%}`)  
**Model Confidence**: **{payload['confidence']:.1%}**  
**Market Regime**: **{payload['market_regime']}** (VIX: `{quant.get('vix_level', 20.0)}`)  
**Risk Level**: **{evidence.risk_summary.get('risk_level', 'MEDIUM')}** (Score: `{evidence.risk_summary.get('risk_score', 0.5):.2f}`)

---

## 1. Executive Summary & Investment Thesis

{payload['analyst_conclusion']}

---

## 2. Quantitative ML Model Signal

- **Signal**: `{payload['signal']}`
- **Probability of Positive 5-Day Return**: `{payload['probability']:.2%}`
- **Decision Threshold (Sharpe-Optimized)**: `{payload['threshold']:.2%}`
- **Edge Above Hurdle**: `{(payload['probability'] - payload['threshold']):+.2%}`
- **Underlying Engine**: `{evidence.quant_features.get('model', 'Random Forest')} (43 engineered features, 25-asset cross-sectional universe)`

---

## 3. SHAP Feature Attribution Analysis

Primary feature contributions identified via TreeSHAP explainability:

### Positive Tailwinds:
{pos_drivers_md if pos_drivers_md else "- None detected"}

### Negative Headwinds:
{neg_drivers_md if neg_drivers_md else "- None detected"}

---

## 4. Historical Analog Pattern Matching (FAISS 30k Engine)

Vector similarity search queried against **30,025 historical observations** across 25 tickers (2021–2026):

- **Sample Size**: `{analogs.get('sample_size', 0)} setups`
- **Historical 5-Day Win Rate**: `{analogs.get('success_rate', 0.0):.1%}`
- **Median Forward Return**: `{analogs.get('median_return_5d', 0.0):+.2%}`
- **Average Forward Return**: `{analogs.get('avg_return_5d', 0.0):+.2%}`
- **95% Confidence Interval**: `[{ci[0]:+.2%}, {ci[1]:+.2%}]`
- **Regime Distribution**: `{analogs.get('market_regime_distribution', {})}`
- **Sector Distribution**: `{analogs.get('sector_distribution', {})}`

### Nearest Historical Precedents:
{top_matches_md}

---

## 5. Technical Indicator & Momentum Setup

- **RSI (14)**: `{tech.get('rsi', 50.0)}` ({tech.get('rsi_condition', 'NEUTRAL')}) — **Cross-Sectional Rank**: `{tech.get('rsi_rank', 0.5):.0%}`
- **MACD Structure**: `{tech.get('macd_condition', 'NEUTRAL')}` (Diff: `{tech.get('macd_diff', 0.0):+.3f}`)
- **Trend Orientation**: `{tech.get('trend', 'BULLISH')}` (EMA20: `{tech.get('ema20')}` vs EMA50: `{tech.get('ema50')}`)
- **Bollinger Squeeze**: `{tech.get('squeeze_state', 'VOLATILITY_EXPANDING')}` (Score: `{tech.get('bb_squeeze', 0.0)}`)
- **Volume Ratio**: `{tech.get('volume_ratio', 1.0)}x` ({tech.get('volume_condition', 'NORMAL')})

---

## 6. Market Regime & Cross-Sectional Ranking

- **Current Regime**: `{payload['market_regime']}`
- **VIX Level**: `{quant.get('vix_level', 20.0)}`
- **Sentiment Cross-Sectional Rank**: `{quant.get('sentiment_rank', 0.5):.0%}` (vs 25 stocks in coverage)
- **Sector 5-Day Momentum**: `{quant.get('sector_return_5d', 0.0):+.2%}`

---

## 7. Key Risks & Invalidation Triggers

{risks_md}

---

## 8. Multi-Hop Knowledge Graph Cascades

{graph_md}

---

*Report synthesized autonomously by the AI Financial Research Assistant (Sprint 4.1 Architecture).*
"""
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(content)



def main():
    parser = argparse.ArgumentParser(description="Sprint 4 AI Financial Research Report Generator")
    parser.add_argument("--ticker", type=str, default="NVDA", help="Stock ticker symbol (default: NVDA)")
    args = parser.parse_args()

    generator = ResearchReportGenerator()
    result = generator.generate_report(args.ticker)

    # Print clean formatted JSON response for consumption
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
