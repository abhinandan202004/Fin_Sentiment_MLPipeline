"""
CLI Script for Phase 6: Committee Evaluation & Counterfactual Analysis.
Sprint 9: Autonomous Evaluation & Adaptive Intelligence.

Evaluates Committee vs ML-Only vs Opportunity-Only vs Benchmarks and generates:
reports/evaluation_reports/committee_performance.md
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.committee_evaluator import CommitteeEvaluator

def main():
    print("\n=======================================================")
    print(" SPRINT 9: COMMITTEE & COUNTERFACTUAL EVALUATION")
    print("=======================================================\n")

    evaluator = CommitteeEvaluator()
    results = evaluator.evaluate_committee_counterfactuals(persist_to_db=True)
    report_file = evaluator.generate_committee_report(results)

    c = results["committee"]
    m = results["ml_only"]
    v = results["veto_analysis"]
    edge = results["edge"]

    print(f"Decisions Evaluated:     {results['decisions_evaluated']}")
    print(f"Committee Return:        {c['annualized_return']:+.2%} (Sharpe: {c['sharpe_ratio']:.2f}, MaxDD: {c['max_drawdown']:.2%})")
    print(f"ML-Only Return:          {m['annualized_return']:+.2%} (Sharpe: {m['sharpe_ratio']:.2f}, MaxDD: {m['max_drawdown']:.2%})")
    print(f"Edge vs ML-Only:         {edge['edge_vs_ml']:+.2%}")
    print(f"Edge vs Benchmark:       {edge['edge_vs_benchmark']:+.2%}")
    print(f"Vetoes Evaluated:        {v['vetoes_evaluated']}")
    print(f"Veto Loss Prevention:    {v['vetoes_prevented_losses']} / {v['vetoes_evaluated']} ({v['veto_loss_prevention_rate']:.1%})")
    print(f"Capital Saved by Vetoes: {v['total_capital_saved_pct']:+.2%}")
    print(f"\nReport Generated:        {report_file}")
    print("=======================================================\n")

if __name__ == "__main__":
    main()
