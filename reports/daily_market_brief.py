"""
Daily Market Intelligence Brief Generator for Sprint 8:
Autonomous Research & Market Opportunity Discovery.

Compiles an institutional daily intelligence brief:
  1. Executive Market Pulse & Macro Regime
  2. Top Ranked Opportunities (Priority Queue)
  3. High-Impact Corporate Catalysts
  4. Secular Macro & Industry Trends
  5. Top Market Risks & Vulnerabilities
  6. Scanner Track Record & Accuracy Attribution (from Outcome Tracker)

Outputs to:
  reports/daily_market_brief.md

Usage:
    python reports/daily_market_brief.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
from typing import Dict, List, Any
from datetime import datetime, timezone

from config import REPORTS_DIR
from research.prioritizer import ResearchPrioritizer
from opportunities.outcome_tracker import OpportunityOutcomeTracker
from agents.trend_agent import TrendAgent
from agents.catalyst_agent import CatalystAgent
from agents.regime_agent import RegimeAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("DailyMarketBrief")


def generate_daily_market_brief() -> str:
    """
    Executes the autonomous discovery pipeline, gathers trend and catalyst intelligence,
    fetches outcome tracking track records, and generates reports/daily_market_brief.md.
    """
    logger.info("=" * 65)
    logger.info("Generating Daily Market Intelligence Brief (Sprint 8)...")
    logger.info("=" * 65)

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # 1. Market Regime
    regime_agent = RegimeAgent()
    reg_res = regime_agent.detect_regime()
    regime = reg_res.get("regime", "Neutral")
    confidence = reg_res.get("confidence", 0.75)
    vix = reg_res.get("vix", 18.5)

    # 2. Research Prioritizer Queue
    prioritizer = ResearchPrioritizer()
    priority_res = prioritizer.generate_prioritized_queue(top_n=10)
    top_opportunities = priority_res.get("top_priority", [])
    full_queue = priority_res.get("full_queue", [])

    # 3. Macro Trends
    trend_agent = TrendAgent()
    trends = trend_agent.analyze_themes()

    # 4. Catalysts
    catalyst_agent = CatalystAgent()
    catalysts = catalyst_agent.scan_all_catalysts(days_back=14, save_to_db=False)

    # 5. Opportunity Outcome Tracker Performance
    outcome_tracker = OpportunityOutcomeTracker()
    perf = outcome_tracker.get_scanner_performance_summary()

    # Build Markdown Document
    md = f"""# Daily Market Intelligence Brief
**Generated:** {now_utc}  
**System:** Fin_Sentiment_MLPipeline — Sprint 8: Autonomous Research & Opportunity Discovery  
**Universe:** 25 Multi-Sector Liquid Equities  

---

## 1. Executive Market Pulse & Macro Regime

| Metric | Current Reading | Interpretation |
|--------|:---------------:|:--------------|
| **Market Regime** | **`{regime.upper()}`** | Systematic macro regime classification |
| **Regime Confidence** | **{confidence:.1%}** | Statistical certainty from volatility/trend features |
| **VIX Index** | **{vix:.2f}** | {'Subdued / Complacent' if vix < 15 else 'Moderate Volatility' if vix < 23 else 'Elevated Stress'} |
| **Opportunity Breadth** | **{len([o for o in top_opportunities if o['composite_score'] >= 70.0])} Assets >= 70** | High-conviction setups currently actionable |

> 💡 **Macro Takeaway**: Market condition is classified as **{regime}**. Capital deployment is calibrated with appropriate risk-adjusted posture.

---

## 2. Top Autonomous Market Opportunities (Priority Queue)

The platform autonomously scanned the coverage universe and ranked the highest-conviction ideas:

| Priority | Ticker | Composite Score | Base Score | Catalyst Boost | Trend Theme | Action |
|:--------:|:------:|:---------------:|:----------:|:--------------:|:------------|:-------|
"""
    for opp in top_opportunities[:8]:
        cat_str = f"{opp['catalyst_boost']:+.1f}" if opp['catalyst_boost'] != 0 else "0.0"
        md += f"| **#{opp['priority']}** | **{opp['ticker']}** | **{opp['composite_score']:.1f}** | {opp['base_opportunity_score']:.1f} | {cat_str} | {opp['trend_theme']} | `{opp['action']}` |\n"

    md += """
### Primary Thesis for Top Picks
"""
    for opp in top_opportunities[:3]:
        md += f"""- **{opp['ticker']} (Score: {opp['composite_score']:.1f})**:
  - **Conviction Metrics**: ML Conviction Probability: `{opp['ml_probability']:.1%}` | FAISS Analog Win Rate: `{opp['analog_win_rate']:.1%}`
  - **Thematic Tailwinds**: Aligned with `{opp['trend_theme']}` secular expansion.
  - **Trigger Catalyst**: {opp['top_catalyst']}
  - **Recommended Workflow**: `{opp['action']}`
"""

    md += """
---

## 3. High-Impact Corporate Catalysts

Recent corporate events and announcements scored by expected price impact:

| Ticker | Catalyst Category | Expected Impact | Headline / Description |
|:------:|:------------------|:---------------:|:-----------------------|
"""
    if catalysts:
        for c in catalysts[:6]:
            imp_str = f"**{c['expected_impact']:+.2f}**"
            desc = (c['description'][:60] + "...") if len(c['description']) > 60 else c['description']
            md += f"| **{c['ticker']}** | {c['catalyst_type']} | {imp_str} | {desc} |\n"
    else:
        md += "| — | *No high-impact catalysts detected in lookback window* | — | — |\n"

    md += """
---

## 4. Secular Macro & Industry Trends Leaderboard

Real-time tracking of institutional themes by news velocity, sentiment polarity, and market momentum:

| Theme | Mention Growth (7d vs 30d) | Sentiment Polarity | News Volume | Key Beneficiary Assets |
|:------|:--------------------------:|:------------------:|:-----------:|:-----------------------|
"""
    for t in trends[:6]:
        growth_str = f"{t['mention_growth']:+.1%}"
        sent_str = f"{t['sentiment_trend']:+.2f}"
        tickers_str = ", ".join(f"**{tk}**" for tk in t['leading_tickers'][:3])
        md += f"| **{t['theme']}** | {growth_str} | {sent_str} | {t['news_volume']} articles | {tickers_str} |\n"

    md += """
---

## 5. Top Market Risks & Vulnerabilities

Assets exhibiting the lowest opportunity scores, negative sentiment drift, or adverse catalyst overhang:

| Ticker | Composite Score | Vulnerability Drivers | Primary Risk Factor |
|:------:|:---------------:|:---------------------|:--------------------|
"""
    low_ranked = sorted(full_queue, key=lambda x: x["composite_score"])[:4]
    for low in low_ranked:
        md += f"| **{low['ticker']}** | {low['composite_score']:.1f} | ML prob `{low['ml_probability']:.1%}`, Low Analog Win Rate | {low['top_catalyst']} |\n"

    md += """
---

## 6. Scanner Track Record & Accuracy Attribution (Outcome Tracker)

Closing the autonomous loop by evaluating realized forward returns of past discoveries:

"""
    if perf.get("status") == "computed" and perf.get("total_resolved", 0) > 0:
        p5_str = f"{perf['precision_at_5']:.1%}" if perf.get("precision_at_5") is not None else "N/A"
        p10_str = f"{perf['precision_at_10']:.1%}" if perf.get("precision_at_10") is not None else "N/A"

        md += f"""| Metric | Value |
|--------|-------|
| **Total Evaluated Discoveries** | {perf['total_resolved']} |
| **Top 5 Precision (Hit Rate)** | **{p5_str}** |
| **Top 10 Precision (Hit Rate)** | **{p10_str}** |

### Forward Returns by Opportunity Score Tier

| Conviction Tier | Sample Count | Avg 5-Day Return | Avg 20-Day Return | Win Rate |
|:----------------|:------------:|:----------------:|:-----------------:|:--------:|
"""
        tiers = perf.get("tier_performance", {})
        for tier_name, stats in tiers.items():
            if "80" in tier_name:
                t_label = "High Conviction (Score >= 80)"
            elif "low" in tier_name or "sub" in tier_name:
                t_label = "Low Conviction (< 70)"
            else:
                t_label = "Medium Conviction (70 - 79)"
            r5 = f"{stats['avg_return_5d']:+.2%}" if stats.get("avg_return_5d") is not None else "—"
            r20 = f"{stats['avg_return_20d']:+.2%}" if stats.get("avg_return_20d") is not None else "—"
            wr = f"{stats['win_rate']:.1%}" if stats.get("win_rate") is not None else "—"
            md += f"| **{t_label}** | {stats['count']} | {r5} | {r20} | {wr} |\n"
    else:
        md += """> ℹ️ *Outcome tracker initializing: 5d, 20d, and 60d forward returns will populate automatically as historical discovery cohorts mature.*
"""

    md += f"""
---

*Report autonomously generated by Fin_Sentiment_MLPipeline — Sprint 8: Autonomous Research & Market Opportunity Discovery.*  
*Institutional Briefing Published: {date_str}*
"""

    # Save to reports directory
    out_file = REPORTS_DIR / "daily_market_brief.md"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(md, encoding="utf-8")
    logger.info(f"Daily Market Intelligence Brief successfully written to: {out_file}")

    return md


if __name__ == "__main__":
    report_md = generate_daily_market_brief()
    print("\n" + "=" * 65)
    print("Daily Market Intelligence Brief Generated Successfully!")
    print(f"Path: reports/daily_market_brief.md")
    print("=" * 65)
