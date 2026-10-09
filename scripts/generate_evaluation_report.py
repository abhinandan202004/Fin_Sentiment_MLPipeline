"""
CLI Script for Phase 13 & 14: Master Evaluation Report Generation.
Sprint 9: Autonomous Evaluation & Adaptive Intelligence.

Executes complete platform evaluation, persists EvaluationSnapshot to database,
and generates reports/evaluation_reports/sprint9_evaluation_report.md.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from evaluation.performance_report import MasterPerformanceReporter

def main():
    parser = argparse.ArgumentParser(description="Sprint 9 Master Evaluation Report Generator")
    parser.add_argument("--initial-capital", type=float, default=1_000_000.0, help="Initial portfolio capital in INR (default: 10,00,000)")
    args = parser.parse_args()

    print("\n=======================================================")
    print(" SPRINT 9: GENERATING MASTER EVALUATION REPORT")
    print("=======================================================\n")

    reporter = MasterPerformanceReporter(initial_capital=args.initial_capital)
    data = reporter.run_full_evaluation(persist_snapshot=True)
    report_file = reporter.generate_master_markdown_report(data)

    pt = data["paper_trading_metrics"]
    cm = data["committee_metrics"]

    print("\n" + "=" * 65)
    print(" SPRINT 9 EVALUATION EXECUTION COMPLETE")
    print("=" * 65)
    print(f"Master Snapshot Persisted: Database evaluation_snapshots")
    print(f"Paper Portfolio Final:     INR {pt['final_equity']:,.2f} ({pt['cumulative_return']:+.2%}, Alpha: {pt['alpha']:+.2%})")
    print(f"Committee Veto Hit Rate:   {cm.get('veto_analysis', {}).get('veto_loss_prevention_rate', 0.0):.1%} ({cm.get('veto_analysis', {}).get('vetoes_prevented_losses', 0)} losses prevented)")
    print(f"Master Report Path:        {report_file}")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    main()
