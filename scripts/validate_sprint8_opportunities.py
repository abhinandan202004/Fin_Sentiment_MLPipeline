"""
Comprehensive Verification Suite for Sprint 8:
Autonomous Research & Market Opportunity Discovery (with Outcome Tracking).

Validates:
1. Database Schema Layer:
   - Verifies tables opportunity_history, trend_history, catalyst_history.
2. Opportunity Scoring Engine (opportunities/scoring.py):
   - Multi-factor calculation, weight normalization, hurdle clearance, bounds [0, 100].
3. Opportunity Scanner (opportunities/scanner.py):
   - Multi-factor scanning across the 25-ticker coverage universe, ranking, and DB persistence.
4. Trend Agent (agents/trend_agent.py):
   - 8 secular themes, mention growth rates, sentiment polarity, leading ticker mapping.
5. Catalyst Agent (agents/catalyst_agent.py):
   - 6 catalyst categories, signed impact scoring [-1.0, +1.0], pattern matching.
6. Research Prioritizer (research/prioritizer.py):
   - Fusion of opportunity score + catalyst boost + trend alignment, actionable queue.
7. Opportunity Outcome Tracker (opportunities/outcome_tracker.py):
   - Multi-horizon forward returns (5d, 20d, 60d), success determination, Precision@5/10.
8. Daily Market Brief (reports/daily_market_brief.py):
   - End-to-end report generation and markdown structure completeness.
"""

import sys
from pathlib import Path
import random
from datetime import datetime, timedelta, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database.connection import SessionLocal, init_db, engine
from database.models import OpportunityHistory, TrendHistory, CatalystHistory
from opportunities.scoring import OpportunityScorer
from opportunities.scanner import OpportunityScanner
from opportunities.outcome_tracker import OpportunityOutcomeTracker
from agents.trend_agent import TrendAgent
from agents.catalyst_agent import CatalystAgent
from research.prioritizer import ResearchPrioritizer
from reports.daily_market_brief import generate_daily_market_brief


def test_schema_and_models():
    print("\n--- 1. Testing Database Schema & Models for Sprint 8 ---")
    init_db()
    session = SessionLocal()
    try:
        # Check querying each model
        opp_count = session.query(OpportunityHistory).count()
        trend_count = session.query(TrendHistory).count()
        cat_count = session.query(CatalystHistory).count()

        print(f"  [PASS] Tables verified: opportunity_history (rows: {opp_count}), trend_history (rows: {trend_count}), catalyst_history (rows: {cat_count})")
    finally:
        session.close()


def test_opportunity_scoring_engine():
    print("\n--- 2. Testing Opportunity Scoring Engine ---")
    scorer = OpportunityScorer()

    # 1. ML Scoring
    score_above = scorer.score_ml(0.68, threshold=0.60)
    score_hurdle = scorer.score_ml(0.60, threshold=0.60)
    score_below = scorer.score_ml(0.48, threshold=0.60)
    assert score_above > score_hurdle > score_below
    assert score_hurdle == 60.0, f"Expected 60.0 at hurdle, got {score_hurdle}"
    print(f"  [PASS] ML Scoring: Above Hurdle = {score_above:.1f}, At Hurdle = {score_hurdle:.1f}, Below = {score_below:.1f}")

    # 2. Analog Scoring
    score_high_wr = scorer.score_analogs(win_rate=0.70, sample_size=50, ci_lower=0.01)
    score_small_n = scorer.score_analogs(win_rate=0.70, sample_size=10, ci_lower=0.01)
    assert score_high_wr > score_small_n, "Small sample size should be penalized"
    print(f"  [PASS] Analog Scoring: Large Sample = {score_high_wr:.1f} vs Small Sample = {score_small_n:.1f}")

    # 3. Composite Calculation
    res = scorer.compute_opportunity(
        ticker="JPM",
        ml_prob=0.65,
        ml_threshold=0.60,
        analog_win_rate=0.68,
        analog_sample_size=50,
        analog_ci_lower=0.005,
        sentiment_rank=0.75,
        sector_return_5d=0.025,
        rsi=56.0,
        squeeze_state="IN_SQUEEZE",
        event_impact=0.40,
        has_recent_catalyst=True,
        market_regime="Bull"
    )
    score = res["opportunity_score"]
    conf = res["confidence"]
    assert 0.0 <= score <= 100.0, f"Score {score} out of bounds"
    assert 0.0 <= conf <= 1.0, f"Confidence {conf} out of bounds"
    assert score >= 70.0, f"High-conviction Bull setup should score >= 70, got {score}"
    print(f"  [PASS] Composite Opportunity Score for JPM: {score} (Confidence: {conf:.1%})")


def test_trend_agent():
    print("\n--- 3. Testing Macro & Thematic Trend Agent ---")
    agent = TrendAgent()
    themes = agent.analyze_themes()
    assert len(themes) >= 6, f"Expected at least 6 themes, got {len(themes)}"

    # Check structure
    theme_names = [t["theme"] for t in themes]
    assert "AI & Machine Learning" in theme_names
    assert "Semiconductors" in theme_names

    top_theme = themes[0]
    assert "mention_growth" in top_theme
    assert "leading_tickers" in top_theme
    assert len(top_theme["leading_tickers"]) > 0

    print(f"  [PASS] Fastest-growing theme detected: '{top_theme['theme']}' (Growth: {top_theme['mention_growth']:+.1%}, Leading: {', '.join(top_theme['leading_tickers'][:3])})")

    # Snapshot to DB
    saved = agent.snapshot_trends(save_to_db=True)
    assert len(saved) == len(themes)
    print(f"  [PASS] Persisted {len(saved)} trend snapshots to trend_history.")


def test_catalyst_agent():
    print("\n--- 4. Testing Corporate Catalyst Agent ---")
    agent = CatalystAgent()

    # Pattern classification
    c_earn = agent.classify_text("JPMorgan reports Q1 earnings revenue beat and raises full year guidance", sentiment_score=0.8)
    assert c_earn is not None
    assert c_earn["catalyst_type"] == "Earnings & Guidance"
    assert c_earn["expected_impact"] > 0.0

    c_anti = agent.classify_text("DOJ opens formal antitrust probe into tech giant practices", sentiment_score=-0.7)
    assert c_anti is not None
    assert c_anti["catalyst_type"] == "Regulation & Legal"
    assert c_anti["expected_impact"] < 0.0

    c_prod = agent.classify_text("NVIDIA unveils next-gen Blackwell GPU architecture at GTC keynote", sentiment_score=0.85)
    assert c_prod is not None
    assert c_prod["catalyst_type"] == "Product Launches"
    assert c_prod["expected_impact"] > 0.0

    print(f"  [PASS] Catalyst Pattern Matching: Earnings = {c_earn['expected_impact']:+.2f}, Antitrust = {c_anti['expected_impact']:+.2f}, Keynote = {c_prod['expected_impact']:+.2f}")

    # Scan universe
    all_cats = agent.scan_all_catalysts(days_back=30, save_to_db=True)
    print(f"  [PASS] Scanned and persisted {len(all_cats)} catalysts across universe.")


def test_opportunity_scanner():
    print("\n--- 5. Testing Opportunity Scanner (Full Universe) ---")
    scanner = OpportunityScanner()
    report = scanner.scan_universe(save_to_db=True)

    assert "opportunities" in report
    opps = report["opportunities"]
    assert len(opps) == 25, f"Expected 25 opportunities scanned, got {len(opps)}"

    # Check sorting
    scores = [o["opportunity_score"] for o in opps]
    assert scores == sorted(scores, reverse=True), "Opportunities must be sorted descending by score"

    # Priority assignment
    priorities = [o["priority"] for o in opps]
    assert priorities == list(range(1, 26)), "Priorities should be sequentially 1 to 25"

    top = opps[0]
    print(f"  [PASS] Scanner successfully ranked 25 tickers. Top Opportunity: {top['ticker']} (Score: {top['opportunity_score']:.1f}, Confidence: {top['confidence']:.1%})")


def test_research_prioritizer():
    print("\n--- 6. Testing Research Prioritizer (Queue & Boosts) ---")
    prioritizer = ResearchPrioritizer()
    res = prioritizer.generate_prioritized_queue(top_n=5)

    assert "top_priority" in res
    top_queue = res["top_priority"]
    assert len(top_queue) == 5, f"Expected 5 items in top queue, got {len(top_queue)}"

    top_item = top_queue[0]
    assert "composite_score" in top_item
    assert "catalyst_boost" in top_item
    assert "trend_boost" in top_item
    assert "action" in top_item

    print(f"  [PASS] Priority #1: {top_item['ticker']} (Composite: {top_item['composite_score']:.1f}, Base: {top_item['base_opportunity_score']:.1f}, Catalyst: {top_item['catalyst_boost']:+.1f}, Trend: {top_item['trend_boost']:+.1f})")
    print(f"         Recommended Action: {top_item['action']}")


def test_opportunity_outcome_tracker():
    print("\n--- 7. Testing Opportunity Outcome Tracker (5d/20d/60d Horizons) ---")
    tracker = OpportunityOutcomeTracker()

    # Seed simulated historical opportunity cohort to test attribution
    session = SessionLocal()
    today = datetime.now(timezone.utc).date()
    random.seed(42)

    try:
        # Check if we already have resolved opportunities
        existing_resolved = session.query(OpportunityHistory).filter(OpportunityHistory.return_5d.isnot(None)).count()
        if existing_resolved < 20:
            print(f"  [INFO] Seeding 25 historical opportunity records for outcome validation...")
            tickers = ["NVDA", "JPM", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "XOM", "CAT", "LLY"]

            for i in range(25):
                t = random.choice(tickers)
                score = round(random.uniform(55.0, 92.0), 1)
                is_high_score = score >= 75.0
                win_prob = 0.72 if is_high_score else 0.42
                is_win = random.random() < win_prob

                ret_5d = round(random.uniform(0.01, 0.05) if is_win else random.uniform(-0.04, -0.005), 4)
                ret_20d = round(ret_5d + (random.uniform(0.01, 0.04) if is_win else random.uniform(-0.03, 0.01)), 4)
                ret_60d = round(ret_20d + random.uniform(-0.02, 0.05), 4)
                max_dd = round(random.uniform(-0.025, -0.008) if is_win else random.uniform(-0.07, -0.02), 4)

                opp = OpportunityHistory(
                    discovery_date=today - timedelta(days=70 - i),
                    ticker=t,
                    score=score,
                    confidence=0.75,
                    ml_probability=0.65 if is_high_score else 0.48,
                    analog_win_rate=0.68 if is_high_score else 0.44,
                    sentiment_rank=0.80 if is_high_score else 0.40,
                    sector_momentum=0.03 if is_high_score else -0.01,
                    event_impact=0.30 if is_high_score else -0.10,
                    market_regime="Bull",
                    priority=(i % 10) + 1,
                    status="RESOLVED",
                    return_5d=ret_5d,
                    return_20d=ret_20d,
                    return_60d=ret_60d,
                    max_drawdown=max_dd,
                    success=bool(ret_20d > 0.0),
                    resolved_date=today
                )
                session.add(opp)
            session.commit()

        # Compute scanner performance summary
        perf = tracker.get_scanner_performance_summary()
        assert perf["status"] == "computed"
        assert perf["total_resolved"] >= 20
        assert perf["precision_at_5"] is not None
        assert perf["precision_at_10"] is not None

        tiers = perf["tier_performance"]
        assert "high_conviction_80_plus" in tiers

        high_wr = tiers["high_conviction_80_plus"].get("win_rate")
        low_wr = tiers["low_conviction_sub_70"].get("win_rate")

        print(f"  [PASS] Opportunity Outcome Tracker: Total Resolved = {perf['total_resolved']}")
        print(f"         Precision@5:  {perf['precision_at_5']:.1%}")
        print(f"         Precision@10: {perf['precision_at_10']:.1%}")
        if high_wr is not None and low_wr is not None:
            print(f"         Monotonicity: High-Score (>=80) Win Rate = {high_wr:.1%} vs Low-Score (<70) = {low_wr:.1%}")

    finally:
        session.close()


def test_daily_market_brief_generation():
    print("\n--- 8. Testing Daily Market Intelligence Brief End-to-End ---")
    report_md = generate_daily_market_brief()
    out_path = Path("reports/daily_market_brief.md")
    assert out_path.exists(), f"Report file {out_path} was not created"

    content = out_path.read_text(encoding="utf-8")
    assert "# Daily Market Intelligence Brief" in content
    assert "## 1. Executive Market Pulse & Macro Regime" in content
    assert "## 2. Top Autonomous Market Opportunities" in content
    assert "## 3. High-Impact Corporate Catalysts" in content
    assert "## 4. Secular Macro & Industry Trends Leaderboard" in content
    assert "## 5. Top Market Risks & Vulnerabilities" in content
    assert "## 6. Scanner Track Record & Accuracy Attribution" in content

    print(f"  [PASS] Daily Market Intelligence Brief generated successfully at:\n         {out_path}")


if __name__ == "__main__":
    print("=" * 75)
    print("Sprint 8 Verification Suite: Autonomous Research & Opportunity Discovery")
    print("=" * 75)

    test_schema_and_models()
    test_opportunity_scoring_engine()
    test_trend_agent()
    test_catalyst_agent()
    test_opportunity_scanner()
    test_research_prioritizer()
    test_opportunity_outcome_tracker()
    test_daily_market_brief_generation()

    print("\n" + "=" * 75)
    print("ALL SPRINT 8 VERIFICATION SUITE TESTS PASSED!")
    print("=" * 75)
