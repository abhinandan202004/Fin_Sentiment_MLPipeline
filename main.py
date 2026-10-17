"""
Unified CLI Entrypoint for Fin_Sentiment_MLPipeline.

Commands:
  dashboard    - Launch the Command Center Web Dashboard and Ops HTTP server
  run-cycle    - Execute the autonomous daily paper-trading cycle
  status       - Inspect system health, readiness, ledger cash, and kill-switch state
  kill-switch  - Activate, deactivate, or inspect the central risk kill-switch
  validate     - Run the comprehensive Sprint 10 production readiness release gate
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

# Auto-reexec into project .venv if run with global python
_root = Path(__file__).resolve().parent
_venv_py_win = _root / ".venv" / "Scripts" / "python.exe"
_venv_py_posix = _root / ".venv" / "bin" / "python"
_venv_py = _venv_py_win if _venv_py_win.exists() else (_venv_py_posix if _venv_py_posix.exists() else None)

if _venv_py and Path(sys.executable).resolve() != _venv_py.resolve() and "PYTHON_VENV_EXEC" not in os.environ:
    env = os.environ.copy()
    env["PYTHON_VENV_EXEC"] = "1"
    res = subprocess.run([str(_venv_py)] + sys.argv, cwd=str(_root), env=env)
    sys.exit(res.returncode)

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Configure clean logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("FinSentimentCLI")


def cmd_dashboard(args):
    """Starts the Command Center HTTP server."""
    from monitoring.api import serve
    print(f"\n========================================================")
    print(f"[*] FIN_SENTIMENT // QUANT COMMAND CENTER")
    print(f"[*] Binding: http://{args.host}:{args.port}")
    print(f"[*] Mode:    SIMULATED PAPER TRADING ONLY")
    print(f"========================================================\n")
    try:
        serve(host=args.host, port=args.port)
    except KeyboardInterrupt:
        print("\nCommand Center stopped by user.")


def cmd_run_cycle(args):
    """Executes the autonomous trading pipeline cycle."""
    from execution.pipeline_runner import TradingPipelineRunner, DEFAULT_UNIVERSE

    cycle_date = None
    if args.date:
        try:
            parts = [int(p) for p in args.date.split("-")]
            cycle_date = date(parts[0], parts[1], parts[2])
        except Exception:
            print(f"Error: Invalid date format '{args.date}', expected YYYY-MM-DD")
            sys.exit(1)

    print(f"\n[*] Launching Autonomous Trading Cycle...")
    if cycle_date:
        print(f"[*] Historical Cycle Date: {cycle_date}")

    runner = TradingPipelineRunner()
    summary = runner.run_cycle(
        universe=DEFAULT_UNIVERSE,
        cycle_date=cycle_date,
        top_k_candidates=args.candidates,
        execute_orders=not args.dry_run
    )

    print(f"\n========================================================")
    print(f"[*] CYCLE COMPLETED: {summary['cycle_id']}")
    print(f"[*] Screened:  {summary['opportunities_screened']} tickers")
    print(f"[*] Reviewed:  {summary['candidates_reviewed']} top candidates")
    print(f"[*] Orders:    {len(summary['orders_submitted'])} submitted, {len(summary['orders_filled'])} filled, {len(summary['orders_rejected'])} rejected")
    print(f"[*] Exits:     {len(summary['positions_closed'])} positions closed")
    
    port = summary.get("portfolio_after", {})
    eq = port.get("total_equity", 12000.0)
    cash = port.get("cash", 12000.0)
    rec_ok = port.get("ledger_reconciled", True)
    delta = port.get("reconcile_delta", 0.0)

    print(f"[*] Equity:    ${eq:,.2f} USD (Cash: ${cash:,.2f})")
    print(f"[*] Invariant: Cash reconciliation {'PASSED (delta=' + str(delta) + ')' if rec_ok else 'FAILED'}")
    print(f"========================================================\n")


def cmd_status(args):
    """Inspects live system health, readiness, and trading status."""
    from monitoring.system_health import SystemHealthMonitor
    from portfolio.risk_guard import RiskGuard
    from execution.order_store import OrderStore
    from portfolio.ledger import PortfolioLedger

    guard = RiskGuard()
    monitor = SystemHealthMonitor(risk_guard=guard)
    active, state = guard.is_kill_switch_active()
    health = monitor.get_health()
    readiness = monitor.get_readiness()

    order_db = Path(__file__).resolve().parent / "data" / "paper_orders.sqlite"
    orders_count = 0
    if order_db.exists():
        st = OrderStore(str(order_db))
        orders_count = len(st.all_orders())
        st.close()

    print(f"\n========================================================")
    print(f"[*] FIN_SENTIMENT SYSTEM STATUS")
    print(f"========================================================")
    print(f"  Health:           {health['status']}")
    print(f"  Readiness:        {'READY' if readiness['ready'] else 'NOT READY'}")
    if not readiness['ready']:
        for issue in readiness.get('gating_issues', []):
            print(f"    - Gating Issue: {issue}")
    print(f"  Trading Mode:     PAPER_TRADING_ONLY")
    print(f"  Kill Switch:      {'ACTIVE (TRADING HALTED)' if active else 'INACTIVE (NORMAL)'}")
    if active:
        print(f"    - Reason:       {state.get('reason')}")
        print(f"    - Triggered By: {state.get('activated_by')}")
    print(f"  Recorded Orders:  {orders_count} in durable SQLite store")
    print(f"========================================================\n")


def cmd_kill_switch(args):
    """Manages the kill-switch."""
    from portfolio.risk_guard import RiskGuard
    guard = RiskGuard()

    if args.action == "status":
        active, state = guard.is_kill_switch_active()
        print(f"\nKill Switch is {'ACTIVE' if active else 'INACTIVE'}")
        if active:
            print(f"Reason:       {state.get('reason')}")
            print(f"Triggered By: {state.get('activated_by')}")
            print(f"Timestamp:    {state.get('activated_at')}")
    elif args.action == "activate":
        reason = args.reason or "Manual emergency CLI activation"
        actor = args.actor or "CLI:Operator"
        state = guard.activate_kill_switch(reason, triggered_by=actor)
        print(f"\n[ALERT] Kill switch ACTIVATED: {reason}")
    elif args.action == "deactivate":
        reason = args.reason or "Manual CLI resumption"
        actor = args.actor or "CLI:Operator"
        state = guard.deactivate_kill_switch(reason, approved_by=actor)
        print(f"\n[OK] Kill switch DEACTIVATED: Normal trading resumed.")


def cmd_validate(args):
    """Runs the release readiness validation suite."""
    from scripts.validate_sprint10_readiness import run_release_gate
    ok = run_release_gate()
    sys.exit(0 if ok else 1)


def main():
    parser = argparse.ArgumentParser(
        description="Fin_Sentiment_MLPipeline Unified Quantitative Trading Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # dashboard
    p_dash = subparsers.add_parser("dashboard", help="Launch Web Command Center & Ops HTTP API")
    p_dash.add_argument("--host", default="127.0.0.1", help="Host binding (default: 127.0.0.1)")
    p_dash.add_argument("--port", type=int, default=8099, help="Port binding (default: 8099)")

    # run-cycle
    p_run = subparsers.add_parser("run-cycle", help="Execute autonomous trading cycle")
    p_run.add_argument("--date", help="Target simulation date (YYYY-MM-DD)")
    p_run.add_argument("--candidates", type=int, default=5, help="Number of top candidates to evaluate (default: 5)")
    p_run.add_argument("--dry-run", action="store_true", help="Screen without executing paper orders")

    # status
    subparsers.add_parser("status", help="Display system health and portfolio state")

    # kill-switch
    p_ks = subparsers.add_parser("kill-switch", help="Control central risk emergency kill-switch")
    p_ks.add_argument("action", choices=["status", "activate", "deactivate"], help="Action to perform")
    p_ks.add_argument("--reason", help="Event reason for activation/deactivation")
    p_ks.add_argument("--actor", help="Operator / system identity")

    # validate
    subparsers.add_parser("validate", help="Run comprehensive master release gate")

    args = parser.parse_args()

    commands = {
        "dashboard": cmd_dashboard,
        "run-cycle": cmd_run_cycle,
        "status": cmd_status,
        "kill-switch": cmd_kill_switch,
        "validate": cmd_validate,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
