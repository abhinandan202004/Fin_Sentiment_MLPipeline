"""
Committee Performance Report Generator for Sprint 7.

End-to-end pipeline that:
1. Resolves all pending outcomes via OutcomeTracker
2. Computes per-agent attribution (accuracy, Brier score, calibration, drawdown)
3. Computes per-signal-stream reliability
4. Stratifies performance by market regime
5. Computes adaptive (learned) agent weights
6. Runs counterfactual analysis (Committee vs ML / Agents / Consensus)
7. Generates reports/committee_performance_report.md

Usage:
    python committee/generate_performance_report.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import logging
from datetime import datetime, timezone

from config import REPORTS_DIR
from committee.outcome_tracker import OutcomeTracker
from committee.agent_attribution import AgentAttribution
from committee.evidence_attribution import EvidenceAttribution
from committee.regime_analysis import RegimeAnalysis
from committee.adaptive_voting import AdaptiveVoting
from committee.counterfactual import CounterfactualAnalysis

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("PerformanceReport")


def generate_performance_report() -> str:
    """
    Runs the complete Sprint 7 attribution pipeline and writes the
    institutional performance report to reports/committee_performance_report.md.
    """
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    logger.info("=" * 60)
    logger.info("Sprint 7 — Committee Performance Report Generation")
    logger.info("=" * 60)

    # --- 1. Resolve Pending Outcomes ---
    logger.info("\n[1/6] Resolving pending outcomes...")
    tracker = OutcomeTracker()
    resolution = tracker.resolve_all_pending()
    resolution_summary = tracker.get_resolution_summary()
    logger.info(f"Resolution: {resolution_summary}")

    # --- 2. Agent Attribution ---
    logger.info("\n[2/6] Computing agent attribution...")
    agent_attr = AgentAttribution()
    agent_metrics = agent_attr.compute_agent_accuracy()
    agent_trends = agent_attr.get_all_agent_trends()
    best_agent = agent_attr.get_best_agent()

    # --- 3. Evidence Attribution ---
    logger.info("\n[3/6] Computing evidence attribution...")
    evidence_attr = EvidenceAttribution()
    signal_reliability = evidence_attr.compute_signal_reliability()
    agreement_value = evidence_attr.compute_signal_agreement_value()
    best_signal = evidence_attr.get_best_signal()

    # --- 4. Regime Analysis ---
    logger.info("\n[4/6] Computing regime analysis...")
    regime = RegimeAnalysis()
    regime_perf = regime.compute_regime_performance()
    regime_matrix = regime.compute_regime_agent_matrix()

    # --- 5. Adaptive Voting ---
    logger.info("\n[5/6] Computing adaptive weights...")
    adaptive = AdaptiveVoting()
    adaptive_result = adaptive.compute_adaptive_weights()
    weight_history = adaptive.get_weight_history()

    # --- 6. Counterfactual Analysis ---
    logger.info("\n[6/6] Running counterfactual analysis...")
    cf = CounterfactualAnalysis()
    counterfactual = cf.run_full_analysis()

    # === Build Report ===
    report = _build_report(
        now_str=now_str,
        resolution_summary=resolution_summary,
        resolution=resolution,
        agent_metrics=agent_metrics,
        agent_trends=agent_trends,
        best_agent=best_agent,
        signal_reliability=signal_reliability,
        agreement_value=agreement_value,
        best_signal=best_signal,
        regime_perf=regime_perf,
        regime_matrix=regime_matrix,
        adaptive_result=adaptive_result,
        weight_history=weight_history,
        counterfactual=counterfactual
    )

    # Write to file
    report_path = REPORTS_DIR / "committee_performance_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"\nReport saved to: {report_path}")

    return report


def _build_report(**kwargs) -> str:
    """Assembles the full institutional performance report."""
    now_str = kwargs["now_str"]
    resolution_summary = kwargs["resolution_summary"]
    resolution = kwargs["resolution"]
    agent_metrics = kwargs["agent_metrics"]
    agent_trends = kwargs["agent_trends"]
    best_agent = kwargs["best_agent"]
    signal_reliability = kwargs["signal_reliability"]
    agreement_value = kwargs["agreement_value"]
    best_signal = kwargs["best_signal"]
    regime_perf = kwargs["regime_perf"]
    regime_matrix = kwargs["regime_matrix"]
    adaptive_result = kwargs["adaptive_result"]
    weight_history = kwargs["weight_history"]
    counterfactual = kwargs["counterfactual"]

    md = f"""# Committee Performance Attribution Report
**Generated:** {now_str}  
**System:** Fin_Sentiment_MLPipeline — Sprint 7: Committee Learning & Performance Attribution

---

## 1. Outcome Resolution Summary

| Metric | Value |
|--------|-------|
| **Total Decisions** | {resolution_summary.get('total', 0)} |
| **Resolved** | {resolution_summary.get('resolved', 0)} |
| **Pending** | {resolution_summary.get('pending', 0)} |
| **Resolution Rate** | {resolution_summary.get('resolution_rate', 0):.0%} |

"""
    # Resolution details
    if resolution.get("details"):
        md += "### Recent Resolutions\n"
        for d in resolution["details"][:10]:
            status = d.get("status", "UNKNOWN")
            ticker = d.get("ticker", "?")
            if status == "RESOLVED":
                rets = d.get("returns", {})
                md += f"- ✅ **{ticker}**: 5d={rets.get('5d', 0):+.2%}, "
                md += f"10d={rets.get('10d', 'N/A')}, 20d={rets.get('20d', 'N/A')}, "
                md += f"MaxDD={d.get('max_drawdown_5d', 0):.2%}\n"
            elif status == "SKIPPED":
                md += f"- ⏳ **{ticker}**: {d.get('reason', 'Too recent')}\n"
            else:
                md += f"- ❌ **{ticker}**: {d.get('reason', 'Failed')}\n"

    md += """
---

## 2. Agent Attribution — Performance Leaderboard

| Agent | Accuracy | Brier Score | Calibration Error | Avg DD on Miss | Samples | Trend | Status |
|-------|:--------:|:-----------:|:-----------------:|:--------------:|:-------:|:-----:|:------:|
"""
    agent_names = ["EvidenceProsecutor", "RiskOfficer", "BullAnalyst", "BearAnalyst", "PortfolioManager"]
    for name in agent_names:
        m = agent_metrics.get(name, {})
        t = agent_trends.get(name, {})
        acc = f"{m.get('accuracy', 0):.0%}" if m.get('accuracy') is not None else "—"
        brier = f"{m.get('brier_score', 0):.3f}" if m.get('brier_score') is not None else "—"
        cal = f"{m.get('calibration_error', 0):.3f}" if m.get('calibration_error') is not None else "—"
        dd = f"{m.get('avg_drawdown_on_miss', 0):.2%}" if m.get('avg_drawdown_on_miss') is not None else "—"
        n = m.get('sample_size', 0)
        trend = t.get('trend', '—')
        trend_emoji = {"improving": "📈", "degrading": "📉", "stable": "➡️"}.get(trend, "—")
        status = m.get('status', 'baseline')
        md += f"| **{name}** | {acc} | {brier} | {cal} | {dd} | {n} | {trend_emoji} {trend} | {status} |\n"

    if best_agent.get("best_agent"):
        md += f"\n> **Best Agent:** `{best_agent['best_agent']}` — {best_agent.get('accuracy', 0):.0%} accuracy ({best_agent.get('sample_size', 0)} samples)\n"

    md += """
---

## 3. Evidence Stream Reliability

| Signal Stream | Accuracy | Hit Rate | Correct | Incorrect | Avg DD on Miss | Samples |
|---------------|:--------:|:--------:|:-------:|:---------:|:--------------:|:-------:|
"""
    for stream in ["ML_MODEL", "ANALOG_ENGINE", "TECHNICAL", "SENTIMENT"]:
        s = signal_reliability.get(stream, {})
        acc = f"{s.get('accuracy', 0):.0%}" if s.get('accuracy') is not None else "—"
        hr = f"{s.get('hit_rate', 0):.0%}" if s.get('hit_rate') is not None else "—"
        correct = s.get('correct', 0)
        incorrect = s.get('incorrect', 0)
        dd = f"{s.get('avg_drawdown_on_miss', 0):.2%}" if s.get('avg_drawdown_on_miss') is not None else "—"
        n = s.get('sample_size', 0)
        md += f"| **{stream}** | {acc} | {hr} | {correct} | {incorrect} | {dd} | {n} |\n"

    if best_signal.get("best_signal"):
        md += f"\n> **Most Reliable Signal:** `{best_signal['best_signal']}` — {best_signal.get('accuracy', 0):.0%} accuracy\n"

    # Agreement value
    if agreement_value.get("status") == "computed":
        ag = agreement_value["agreement"]
        dg = agreement_value["disagreement"]
        edge = agreement_value.get("agreement_edge", 0)
        md += f"""
### Signal Agreement Value Analysis

| Metric | Agreement Cases | Disagreement Cases |
|--------|:--------------:|:------------------:|
| **Count** | {ag['count']} | {dg['count']} |
| **Avg 5d Return** | {ag['avg_return_5d']:+.2%} | {dg['avg_return_5d']:+.2%} |
| **Median 5d Return** | {ag['median_return_5d']:+.2%} | {dg['median_return_5d']:+.2%} |
| **Avg Drawdown** | {ag['avg_drawdown']:.2%} | {dg['avg_drawdown']:.2%} |
| **Positive Rate** | {ag['positive_rate']:.0%} | {dg['positive_rate']:.0%} |
| **Edge** | **{edge:+.2%}** | — |
"""

    md += """
---

## 4. Regime Performance Analysis

"""
    if regime_perf.get("regimes"):
        md += "| Regime | Decisions | Accuracy | Avg 5d | Avg 10d | Avg 20d | Avg DD | Max DD | Post-Dec Vol |\n"
        md += "|--------|:---------:|:--------:|:------:|:-------:|:-------:|:------:|:------:|:------------:|\n"
        for regime_name, rp in regime_perf["regimes"].items():
            acc = f"{rp['accuracy']:.0%}" if rp.get("accuracy") is not None else "—"
            r5 = f"{rp['avg_return_5d']:+.2%}" if rp.get("avg_return_5d") is not None else "—"
            r10 = f"{rp['avg_return_10d']:+.2%}" if rp.get("avg_return_10d") is not None else "—"
            r20 = f"{rp['avg_return_20d']:+.2%}" if rp.get("avg_return_20d") is not None else "—"
            add = f"{rp['avg_drawdown']:.2%}" if rp.get("avg_drawdown") is not None else "—"
            mdd = f"{rp['max_drawdown']:.2%}" if rp.get("max_drawdown") is not None else "—"
            vol = f"{rp['avg_post_decision_volatility']:.2%}" if rp.get("avg_post_decision_volatility") is not None else "—"
            md += f"| **{regime_name}** | {rp['total_decisions']} | {acc} | {r5} | {r10} | {r20} | {add} | {mdd} | {vol} |\n"
    else:
        md += "> No regime-stratified data available yet.\n"

    # Regime × Agent Matrix
    if regime_matrix.get("matrix"):
        md += "\n### Agent × Regime Accuracy Matrix\n\n"
        all_agents = set()
        for agents in regime_matrix["matrix"].values():
            all_agents.update(agents.keys())
        sorted_agents = sorted(all_agents)

        md += "| Regime | " + " | ".join(f"**{a}**" for a in sorted_agents) + " |\n"
        md += "|--------|" + "|".join(":------:" for _ in sorted_agents) + "|\n"
        for regime_name, agents in regime_matrix["matrix"].items():
            row = f"| **{regime_name}** |"
            for a in sorted_agents:
                data = agents.get(a, {})
                if data.get("accuracy") is not None:
                    row += f" {data['accuracy']:.0%} (n={data['sample_size']}) |"
                else:
                    row += " — |"
            md += row + "\n"

    md += """
---

## 5. Adaptive Voting Weights

"""
    md += f"**Status:** `{adaptive_result['status'].upper()}`  \n"
    md += f"**Resolved Decisions:** {adaptive_result.get('resolved_count', 0)}"
    if adaptive_result["status"] == "baseline":
        md += f" / {adaptive_result.get('threshold', 50)} required\n\n"
    else:
        md += f"\n\n"

    md += "| Agent | Current Weight | EWMA Accuracy | Baseline Weight |\n"
    md += "|-------|:--------------:|:-------------:|:---------------:|\n"
    baseline = {"BullAnalyst": 0.35, "BearAnalyst": 0.25, "RiskOfficer": 0.25, "EvidenceProsecutor": 0.15}
    for agent in ["BullAnalyst", "BearAnalyst", "RiskOfficer", "EvidenceProsecutor"]:
        w = adaptive_result["weights"].get(agent, 0)
        ewma = adaptive_result.get("agent_ewma", {}).get(agent, "—")
        bl = baseline.get(agent, 0)
        ewma_str = f"{ewma:.2%}" if isinstance(ewma, (int, float)) else "—"
        delta = "→" if abs(w - bl) < 0.01 else ("↑" if w > bl else "↓")
        md += f"| **{agent}** | {w:.2%} {delta} | {ewma_str} | {bl:.2%} |\n"

    # Weight history
    if weight_history and len(weight_history) > 1:
        md += "\n### Weight Evolution (Last 5 Checkpoints)\n\n"
        md += "| Checkpoint | BullAnalyst | BearAnalyst | RiskOfficer | EvidenceProsecutor |\n"
        md += "|:----------:|:-----------:|:-----------:|:-----------:|:------------------:|\n"
        for h in weight_history[-5:]:
            w = h["weights"]
            md += f"| @{h['decisions_count']} decisions | {w.get('BullAnalyst', 0):.2%} | {w.get('BearAnalyst', 0):.2%} | {w.get('RiskOfficer', 0):.2%} | {w.get('EvidenceProsecutor', 0):.2%} |\n"

    md += """
---

## 6. Counterfactual Analysis — Is the Committee Worth It?

"""
    if counterfactual.get("status") == "computed":
        cp = counterfactual.get("committee", {})
        ml = counterfactual.get("ml_only", {})
        cs = counterfactual.get("consensus_signals", {})

        md += "### Strategy Comparison\n\n"
        md += "| Strategy | Accuracy | Avg 5d Return | Avg Drawdown | Sharpe Proxy | Decisions |\n"
        md += "|----------|:--------:|:-------------:|:------------:|:------------:|:---------:|\n"

        for label, data in [("**Committee (Actual)**", cp), ("ML Model Only", ml), ("Consensus Signals", cs)]:
            acc = f"{data.get('accuracy', 0):.0%}" if data.get('accuracy') is not None else "—"
            ret = f"{data.get('avg_return_5d', 0):+.2%}" if data.get('avg_return_5d') is not None else "—"
            dd = f"{data.get('avg_drawdown', 0):.2%}" if data.get('avg_drawdown') is not None else "—"
            sharpe = f"{data.get('sharpe_proxy', 0):.2f}" if data.get('sharpe_proxy') is not None else "—"
            n = data.get('sample_size', 0)
            md += f"| {label} | {acc} | {ret} | {dd} | {sharpe} | {n} |\n"

        # Individual agents
        ind_agents = counterfactual.get("individual_agents", {})
        if ind_agents:
            md += "\n### Committee vs Individual Agents\n\n"
            md += "| Agent | Accuracy | Avg 5d Return | Avg Drawdown | Sharpe Proxy |\n"
            md += "|-------|:--------:|:-------------:|:------------:|:------------:|\n"
            for agent_name, data in ind_agents.items():
                acc = f"{data.get('accuracy', 0):.0%}" if data.get('accuracy') is not None else "—"
                ret = f"{data.get('avg_return_5d', 0):+.2%}" if data.get('avg_return_5d') is not None else "—"
                dd = f"{data.get('avg_drawdown', 0):.2%}" if data.get('avg_drawdown') is not None else "—"
                sharpe = f"{data.get('sharpe_proxy', 0):.2f}" if data.get('sharpe_proxy') is not None else "—"
                md += f"| {agent_name} | {acc} | {ret} | {dd} | {sharpe} |\n"

        # Edges
        edges = counterfactual.get("edges", {})
        if edges:
            md += "\n### Edge Analysis\n\n"
            md += "| Comparison | Accuracy Edge | Return Edge | Drawdown Edge | Verdict |\n"
            md += "|------------|:------------:|:-----------:|:-------------:|:-------:|\n"
            for comp, edge in edges.items():
                acc_e = f"{edge.get('accuracy_edge', 0):+.2%}" if edge.get('accuracy_edge') is not None else "—"
                ret_e = f"{edge.get('return_edge', 0):+.2%}" if edge.get('return_edge') is not None else "—"
                dd_e = f"{edge.get('drawdown_edge', 0):+.2%}" if edge.get('drawdown_edge') is not None else "—"
                verdict = edge.get("verdict", "—")
                md += f"| {comp} | {acc_e} | {ret_e} | {dd_e} | `{verdict}` |\n"

        # Overall verdict
        overall = counterfactual.get("overall_verdict", "INCONCLUSIVE")
        md += f"\n### 🏛️ Overall Verdict\n\n> **{overall}**\n"
    else:
        reason = counterfactual.get("reason", "Insufficient data")
        md += f"> ⚠️ Counterfactual analysis not available: {reason}\n"

    md += f"""
---

## 7. System Status

| Component | Status |
|-----------|--------|
| **Outcome Resolution** | {resolution_summary.get('resolved', 0)}/{resolution_summary.get('total', 0)} resolved |
| **Agent Attribution** | {agent_metrics.get('_meta', {}).get('total_resolved_votes', 0)} evaluated votes |
| **Evidence Attribution** | {signal_reliability.get('_meta', {}).get('total_decisions', 0)} decisions analyzed |
| **Adaptive Weights** | `{adaptive_result['status'].upper()}` ({adaptive_result.get('resolved_count', 0)} decisions) |
| **Counterfactual Analysis** | `{counterfactual.get('status', 'N/A').upper()}` |

---

*Report generated by Sprint 7 — Committee Learning & Performance Attribution.*  
*Fin_Sentiment_MLPipeline — {now_str}*
"""
    return md


if __name__ == "__main__":
    report = generate_performance_report()
    print(f"\n{'='*60}")
    print("Report generated successfully!")
    print(f"{'='*60}")
