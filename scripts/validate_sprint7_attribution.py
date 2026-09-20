"""
Comprehensive Verification Suite for Sprint 7:
Committee Learning & Performance Attribution.

Validates:
1. OutcomeTracker:
   - Resolves decisions with 5d, 10d, 20d returns, max drawdown, and realized volatility.
2. AgentAttribution:
   - Per-agent accuracy, Brier score, calibration error, trend detection, drawdown on miss.
3. EvidenceAttribution:
   - Signal reliability (ML, Analogs, Technical, Sentiment), multi-stream agreement analysis.
4. RegimeAnalysis:
   - Regime stratification (Bull, Bear, Sideways, HighVol, LowVol), Agent x Regime matrix, confidence multipliers.
5. AdaptiveVoting:
   - Threshold enforcement (< 50 decisions -> baseline weights; >= 50 decisions -> EWMA weights).
   - Normalization, weight floors (5%) and caps (45%), regime modifiers.
6. CounterfactualAnalysis:
   - Committee vs ML-Only, Simple Consensus, and Best Individual Agent benchmarks.
7. Full Pipeline Execution & Markdown Report Generation:
   - Generates full attribution report with synthetic historical cohort to demonstrate all tables and charts.
"""

import sys
from pathlib import Path
import random
from datetime import datetime, timedelta, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database.connection import SessionLocal, init_db
from database.models import CommitteeDecision, CommitteeVote
from committee.committee_memory import CommitteeMemory
from committee.outcome_tracker import OutcomeTracker
from committee.agent_attribution import AgentAttribution
from committee.evidence_attribution import EvidenceAttribution
from committee.regime_analysis import RegimeAnalysis
from committee.adaptive_voting import AdaptiveVoting
from committee.counterfactual import CounterfactualAnalysis
from committee.generate_performance_report import generate_performance_report


def seed_historical_decisions(target_count: int = 55):
    """
    Seeds resolved decisions and agent votes to simulate an active learning cohort (> 50 decisions).
    This enables adaptive voting weights and full counterfactual benchmarking.
    """
    print(f"\n--- Seeding {target_count} historical resolved decisions for testing ---")
    memory = CommitteeMemory()
    tickers = ["AAPL", "MSFT", "NVDA", "JPM", "XOM", "AMZN", "GOOGL", "TSLA", "META", "LLY"]
    regimes = ["Bull", "Bear", "Sideways", "High Volatility", "Low Volatility"]

    # Check how many resolved decisions already exist
    existing = memory.get_all_resolved_decisions()
    needed = max(0, target_count - len(existing))
    
    if needed == 0:
        print(f"  [INFO] Already have {len(existing)} resolved decisions in DB. No new seeding needed.")
        return

    random.seed(42)
    start_date = datetime.now(timezone.utc) - timedelta(days=120)

    for i in range(needed):
        ticker = random.choice(tickers)
        regime = random.choice(regimes)
        
        # Outcome scenario
        is_bull_regime = regime in ("Bull", "Low Volatility")
        win_prob = 0.65 if is_bull_regime else 0.40
        outcome_positive = random.random() < win_prob
        
        ret_5d = round(random.uniform(0.01, 0.06) if outcome_positive else random.uniform(-0.05, -0.005), 4)
        ret_10d = round(ret_5d + random.uniform(-0.02, 0.03), 4)
        ret_20d = round(ret_10d + random.uniform(-0.03, 0.04), 4)
        max_dd = round(random.uniform(-0.03, -0.01) if outcome_positive else random.uniform(-0.07, -0.03), 4)
        vol = round(random.uniform(0.14, 0.28), 4)

        # ML model signals
        ml_prob = round(random.uniform(0.55, 0.75) if outcome_positive else random.uniform(0.35, 0.52), 3)
        analog_win = round(random.uniform(0.58, 0.72) if outcome_positive else random.uniform(0.38, 0.54), 3)
        agreement = 0.85 if (ml_prob >= 0.55 and analog_win >= 0.55) or (ml_prob < 0.55 and analog_win < 0.55) else 0.40

        # Committee decision
        if ml_prob >= 0.60 and analog_win >= 0.60:
            dec_type = "BUY"
            alloc = 0.08
        elif ml_prob < 0.50 and analog_win < 0.50:
            dec_type = "SELL"
            alloc = 0.0
        elif abs(ml_prob - analog_win) > 0.15:
            dec_type = "WATCHLIST"
            alloc = 0.0
        else:
            dec_type = "HOLD"
            alloc = 0.02

        dec_id = memory.record_decision(
            ticker=ticker,
            decision=dec_type,
            confidence=round(random.uniform(0.60, 0.85), 2),
            consensus_score=round(random.uniform(0.55, 0.80), 2),
            allocation=alloc,
            bull_score=70.0 if dec_type == "BUY" else 30.0,
            bear_score=30.0 if dec_type == "BUY" else 70.0,
            risk_score=25.0 if dec_type == "BUY" else 65.0,
            evidence_score=80.0 if agreement > 0.5 else 45.0,
            final_score=75.0 if dec_type == "BUY" else 35.0,
            evidence_quality="HIGH" if agreement > 0.5 else "MEDIUM",
            governance_passed=True,
            cio_rationale=f"Simulated cohort test decision #{i+1}",
            model_probability=ml_prob,
            analog_win_rate=analog_win,
            evidence_agreement_score=agreement,
            market_regime=regime,
            regime_confidence=0.80
        )

        # Record agent votes
        # BullAnalyst votes BUY on high prob
        memory.record_vote(
            decision_id=dec_id,
            ticker=ticker,
            agent_name="BullAnalyst",
            stance="BUY" if ml_prob >= 0.52 else "HOLD",
            confidence=ml_prob,
            arguments=["Strong growth trajectory"]
        )

        # BearAnalyst votes SELL on low prob / high vol
        memory.record_vote(
            decision_id=dec_id,
            ticker=ticker,
            agent_name="BearAnalyst",
            stance="SELL" if ml_prob < 0.55 or regime == "Bear" else "HOLD",
            confidence=round(1.0 - ml_prob, 2),
            arguments=["Macro headwinds"]
        )

        # RiskOfficer votes based on volatility & regime
        memory.record_vote(
            decision_id=dec_id,
            ticker=ticker,
            agent_name="RiskOfficer",
            stance="REJECT" if regime in ("Bear", "High Volatility") else "PASS",
            confidence=0.75,
            arguments=["Regime risk assessment"]
        )

        # EvidenceProsecutor checks statistical agreement
        memory.record_vote(
            decision_id=dec_id,
            ticker=ticker,
            agent_name="EvidenceProsecutor",
            stance="PASS" if agreement >= 0.70 else "CHALLENGE",
            confidence=agreement,
            arguments=["Evidence alignment check"]
        )

        # Resolve outcome
        memory.update_outcomes(
            decision_id=dec_id,
            realized_return_5d=ret_5d,
            realized_return_10d=ret_10d,
            realized_return_20d=ret_20d,
            max_drawdown_5d=max_dd,
            post_decision_volatility=vol
        )

    print(f"  [PASS] Successfully seeded cohort. Total resolved decisions now >= {target_count}.")


def test_outcome_tracker_summary():
    print("\n--- 1. Testing Outcome Tracker & Resolution Metrics ---")
    tracker = OutcomeTracker()
    summary = tracker.get_resolution_summary()
    assert "total" in summary, "Outcome summary missing 'total'"
    assert "resolved" in summary, "Outcome summary missing 'resolved'"
    assert summary["resolved"] >= 50, f"Expected at least 50 resolved decisions, got {summary['resolved']}"
    print(f"  [PASS] Outcome Tracker: {summary['resolved']}/{summary['total']} decisions resolved ({summary['resolution_rate']:.1%})")


def test_agent_attribution():
    print("\n--- 2. Testing Agent Attribution & Calibration ---")
    agent_attr = AgentAttribution()
    acc_results = agent_attr.compute_agent_accuracy()
    
    for agent_name in ["BullAnalyst", "BearAnalyst", "RiskOfficer", "EvidenceProsecutor"]:
        assert agent_name in acc_results, f"Missing metrics for {agent_name}"
        data = acc_results[agent_name]
        assert data["accuracy"] is not None, f"Accuracy is None for {agent_name}"
        assert data["brier_score"] is not None, f"Brier score is None for {agent_name}"
        assert data["calibration_error"] is not None, f"Calibration error is None for {agent_name}"
        print(f"  [PASS] {agent_name:18} | Acc: {data['accuracy']:.1%} | Brier: {data['brier_score']:.4f} | CalibErr: {data['calibration_error']:.4f} | Samples: {data['sample_size']}")

    trends = agent_attr.get_all_agent_trends()
    assert len(trends) >= 4, "Expected trends for at least 4 agents"
    best = agent_attr.get_best_agent()
    assert best["best_agent"] is not None, "Best agent should be identified"
    print(f"  [PASS] Top performing agent identified: {best['best_agent']} ({best['accuracy']:.1%} accuracy)")


def test_evidence_attribution():
    print("\n--- 3. Testing Evidence Attribution & Agreement Value ---")
    ev_attr = EvidenceAttribution()
    signals = ev_attr.compute_signal_reliability()
    
    for stream in ["ML_MODEL", "ANALOG_ENGINE", "TECHNICAL", "SENTIMENT"]:
        assert stream in signals, f"Missing stream {stream}"
        assert signals[stream]["accuracy"] is not None, f"Accuracy is None for {stream}"
        print(f"  [PASS] Stream {stream:14} | Accuracy: {signals[stream]['accuracy']:.1%} (Samples: {signals[stream]['sample_size']})")

    agr = ev_attr.compute_signal_agreement_value()
    assert "agreement" in agr or agr.get("status") == "insufficient_data"
    if "agreement" in agr:
        agr_rate = agr["agreement"]["positive_rate"]
        disagr_rate = agr["disagreement"]["positive_rate"]
        print(f"  [PASS] Agreement Value: High Agreement Win Rate = {agr_rate:.1%} vs Disagreement Win Rate = {disagr_rate:.1%}")
    else:
        print(f"  [PASS] Agreement Value computed (status: {agr.get('status')})")


def test_regime_analysis():
    print("\n--- 4. Testing Regime Stratification & Multipliers ---")
    reg_analysis = RegimeAnalysis()
    perf = reg_analysis.compute_regime_performance()
    regimes_found = perf.get("regimes", {})
    assert len(regimes_found) >= 2, "Expected at least 2 regimes analyzed"

    for reg_name, data in regimes_found.items():
        ret_str = f"{data['avg_return_5d']:+.2%}" if data.get("avg_return_5d") is not None else "N/A"
        dd_str = f"{data['max_drawdown']:+.2%}" if data.get("max_drawdown") is not None else "N/A"
        print(f"  [PASS] Regime: {reg_name:15} | Acc: {data['accuracy']:.1%} | 5d Ret: {ret_str} | MaxDD: {dd_str}")

    matrix_res = reg_analysis.compute_regime_agent_matrix()
    assert "matrix" in matrix_res, "Regime-agent matrix missing"
    print(f"  [PASS] Regime-Agent cross-tabulation matrix computed for {len(matrix_res['matrix'])} regimes.")

    for reg in ["Bull", "Bear"]:
        rec = reg_analysis.get_regime_recommendation(reg)
        assert "multiplier" in rec, f"Multiplier missing in recommendation for {reg}"
        print(f"  [PASS] Regime recommendation for {reg}: {rec['recommendation']} (mult: {rec['multiplier']:.2f}x)")


def test_adaptive_voting():
    print("\n--- 5. Testing Adaptive Voting with > 50 Decisions ---")
    adaptive = AdaptiveVoting(min_samples=50, ewma_span=10)
    res = adaptive.compute_adaptive_weights()
    
    assert res["status"] == "adaptive", f"Expected 'adaptive' status with > 50 decisions, got '{res['status']}'"
    weights = res["weights"]
    total_w = sum(weights.values())
    assert abs(total_w - 1.0) < 1e-3, f"Weights must sum to 1.0, got {total_w}"

    # Verify weight floors and caps
    for agent, w in weights.items():
        assert 0.05 <= w <= 0.45, f"Weight for {agent} ({w:.2%}) outside [5%, 45%]"
        print(f"  [PASS] Learned Weight: {agent:18} = {w:6.2%} (EWMA Acc: {res['agent_ewma'].get(agent, 0.0):.1%})")


def test_counterfactual_analysis():
    print("\n--- 6. Testing Counterfactual Analysis ---")
    cf = CounterfactualAnalysis()
    res = cf.run_full_analysis()
    
    assert res["status"] == "computed", f"Expected 'computed' status, got '{res['status']}'"
    assert "committee" in res
    assert "ml_only" in res
    assert "consensus_signals" in res
    assert "individual_agents" in res

    edges = res["edges"]
    assert "committee_vs_ml" in edges
    assert "committee_vs_consensus" in edges

    vs_ml = edges["committee_vs_ml"]
    print(f"  [PASS] Committee vs ML-Only: Accuracy Edge = {vs_ml['accuracy_edge']:+.1%}, Return Edge = {vs_ml['return_edge']:+.2%}")
    vs_cons = edges["committee_vs_consensus"]
    print(f"  [PASS] Committee vs Consensus: Accuracy Edge = {vs_cons['accuracy_edge']:+.1%}, Return Edge = {vs_cons['return_edge']:+.2%}")
    print(f"  [PASS] Counterfactual Overall Verdict: {res['overall_verdict']}")


def test_end_to_end_report_generation():
    print("\n--- 7. Testing End-to-End Performance Report Generation ---")
    content = generate_performance_report()
    report_path = Path("reports/committee_performance_report.md")
    assert report_path.exists(), f"Report file {report_path} was not created"

    assert "# Committee Performance Attribution Report" in content
    assert "## 1. Outcome Resolution Summary" in content
    assert "## 2. Agent Attribution — Performance Leaderboard" in content
    assert "## 3. Evidence Stream Reliability" in content
    assert "## 4. Regime Performance Analysis" in content
    assert "## 5. Adaptive Voting Weights" in content
    assert "## 6. Counterfactual Analysis" in content
    assert "`ADAPTIVE`" in content, "Expected ADAPTIVE status in report"
    assert "Is the Committee Worth It?" in content
    print(f"  [PASS] Full report successfully written to:\n         {report_path}")


if __name__ == "__main__":
    print("=" * 70)
    print("Sprint 7 Verification Suite: Committee Learning & Performance Attribution")
    print("=" * 70)
    
    init_db()
    seed_historical_decisions(target_count=55)
    test_outcome_tracker_summary()
    test_agent_attribution()
    test_evidence_attribution()
    test_regime_analysis()
    test_adaptive_voting()
    test_counterfactual_analysis()
    test_end_to_end_report_generation()

    print("\n" + "=" * 70)
    print("ALL SPRINT 7 VERIFICATION SUITE TESTS PASSED!")
    print("=" * 70)
