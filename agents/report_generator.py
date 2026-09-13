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
        signal = ml_pred.get("prediction", "NEUTRAL")
        confidence = float(ml_pred.get("confidence", 0.50))
        top_drivers = ml_pred.get("top_positive_drivers", []) + ml_pred.get("top_negative_drivers", [])

        # 2. Event Detection Engine
        raw_events = self.event_agent.detect_events_for_ticker(ticker, limit=6)
        event_names = [e["event"] for e in raw_events] if raw_events else ["routine_disclosure"]

        # 3. Financial Knowledge Graph
        graph_impacts = self.graph_engine.get_ticker_graph_impacts(ticker)
        if not graph_impacts:
            graph_impacts = [f"Direct equity monitoring for {ticker}"]

        # 4. Historical Analog Search (FAISS)
        analog_results = self.analog_agent.analyze_analogs(ticker, top_k=42)

        # 5. Technical & Risk Assessment
        tech_summary = self.technical_agent.analyze_ticker(ticker)
        risk_summary = self.risk_agent.evaluate_risk(ticker)

        # 6. Build Shared Evidence Contract
        evidence = Evidence(
            ticker=ticker,
            prediction=signal,
            confidence=confidence,
            events=raw_events,
            graph_impacts=graph_impacts,
            analogs=analog_results,
            technical_summary=tech_summary,
            risk_summary=risk_summary,
            shap_drivers=top_drivers,
        )

        # 7. Analyst Agent Synthesis
        conclusion = self.analyst_agent.synthesize(evidence)

        # 8. API Contract Response (Sprint 4 Required Format)
        response_payload = {
            "ticker": ticker,
            "signal": signal,
            "confidence": round(confidence, 2),
            "events": event_names[:3],
            "graph_impacts": [graph_impacts[0]] if graph_impacts else ["Sector correlation"],
            "historical_analogs": {
                "similar_cases": analog_results.get("similar_cases", 42),
                "success_rate": round(float(analog_results.get("success_rate", 0.75)), 2),
            },
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
        for m in analogs.get("top_matches", [])[:3]:
            top_matches_md += f"- `{m['date']}` ({m['ticker']}): 5d Return: `{m['return_5d']:+.2%}` | Event: *{m['event']}* (Distance: {m['distance']})\n"

        risks_md = ""
        for r in evidence.risk_summary.get("risk_factors", []):
            risks_md += f"- {r}\n"

        drivers_md = ""
        for d in evidence.shap_drivers[:5]:
            drivers_md += f"- `{d}`\n"

        content = f"""# Equity Research Note: {ticker}

**Generated**: {now_str}  
**Primary Stance**: **{payload['signal']}** (Model Confidence: **{payload['confidence']:.0%}**)  
**Risk Level**: **{evidence.risk_summary.get('risk_level', 'MEDIUM')}** (Score: `{evidence.risk_summary.get('risk_score', 0.5):.2f}`)

---

## 1. Executive Summary & Investment Thesis

{payload['analyst_conclusion']}

---

## 2. Quantitative ML Model Signal (Sprint 3 Registry)

- **Recommendation**: `{payload['signal']}`
- **Model Confidence**: `{payload['confidence']:.2%}`
- **Top Feature Drivers (TreeSHAP Attributions)**:
{drivers_md}

---

## 3. Detected Corporate & Macro Events (NLP Event Engine)

{events_md}

---

## 4. Multi-Hop Financial Knowledge Graph Cascades

{graph_md}

---

## 5. Historical Analog Pattern Matching (FAISS Vector Engine)

- **Total Highly Correlated Setups**: `{analogs.get('similar_cases', 0)}`
- **Historical 5-Day Win Rate**: `{analogs.get('success_rate', 0.0):.1%}`
- **Median Forward Return**: `{analogs.get('median_return_5d', 0.0):+.2%}`
- **Average Forward Return**: `{analogs.get('avg_return_5d', 0.0):+.2%}`

### Nearest Historical Precedents:
{top_matches_md}

---

## 6. Technical Indicator & Momentum Setup

- **RSI (14)**: `{evidence.technical_summary.get('rsi', 50.0)}` ({evidence.technical_summary.get('rsi_condition', 'NEUTRAL')})
- **MACD Structure**: `{evidence.technical_summary.get('macd_condition', 'NEUTRAL')}` (Diff: `{evidence.technical_summary.get('macd_diff', 0.0):+.3f}`)
- **Trend Orientation**: `{evidence.technical_summary.get('trend', 'BULLISH')}`
- **Volume Ratio**: `{evidence.technical_summary.get('volume_ratio', 1.0)}x` ({evidence.technical_summary.get('volume_condition', 'NORMAL')})

---

## 7. Key Risks & Invalidation Triggers

{risks_md}

---

*Report synthesized autonomously by the AI Financial Research Assistant (Sprint 4 Architecture).*
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
