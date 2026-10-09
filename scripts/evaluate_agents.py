"""
CLI Script for Phase 5: Agent Performance Evaluation.
Sprint 9: Autonomous Evaluation & Adaptive Intelligence.

Evaluates each committee agent independently and produces:
reports/evaluation_reports/agent_performance.md
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.agent_evaluator import AgentEvaluator

def main():
    print("\n=======================================================")
    print(" SPRINT 9: AGENT PERFORMANCE EVALUATION")
    print("=======================================================\n")

    evaluator = AgentEvaluator()
    metrics = evaluator.evaluate_all_agents(persist_to_db=True)
    report_file = evaluator.generate_agent_report(metrics)

    print(f"{'Agent Name':<22} | {'Obs':<5} | {'Hit Rate':<10} | {'ECE':<8} | {'Sharpe':<8} | {'Avg 5d Ret':<12}")
    print("-" * 75)
    for a in sorted(metrics.values(), key=lambda x: x["sharpe_ratio"], reverse=True):
        print(f"{a['agent_name']:<22} | {a['sample_size']:<5} | {a['hit_rate']:<10.1%} | {a['ece']:<8.4f} | {a['sharpe_ratio']:<8.2f} | {a['average_return']:<+12.2%}")

    print("\n" + "=" * 75)
    print(f"Report written to: {report_file}")
    print("=======================================================\n")

if __name__ == "__main__":
    main()
