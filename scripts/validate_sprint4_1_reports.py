import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.report_generator import ResearchReportGenerator

def main():
    generator = ResearchReportGenerator()
    tickers = ["NVDA", "AAPL", "JPM", "XOM"]
    print("=" * 70)
    print("  SPRINT 4.1 VALIDATION SUITE: MULTI-SECTOR RESEARCH REPORTS")
    print("=" * 70)

    for ticker in tickers:
        print(f"\n--- Generating Report: {ticker} ---")
        res = generator.generate_report(ticker, save_markdown=True)
        analogs = res["historical_analogs"]
        shap_pos = res["top_shap_drivers"]["positive"]
        shap_neg = res["top_shap_drivers"]["negative"]
        ci = analogs["confidence_interval"]

        print(f"Ticker: {res['ticker']} | Signal: {res['signal']} | Regime: {res['market_regime']}")
        print(f"Probability: {res['probability']:.2%} | Threshold: {res['threshold']:.2%} | Confidence: {res['confidence']:.2%}")
        print(f"FAISS Analogs ({analogs['sample_size']} cases): Win Rate: {analogs['success_rate']:.1%} | Median 5d: {analogs['median_return_5d']:+.2%} | 95% CI: [{ci[0]:+.2%}, {ci[1]:+.2%}]")
        print(f"Top Positive SHAP: {shap_pos}")
        print(f"Top Negative SHAP: {shap_neg}")
        print(f"Thesis: {res['analyst_conclusion'][:140]}...")

    print("\n" + "=" * 70)
    print("  SPRINT 4.1 VALIDATION COMPLETE: ALL 4 SECTORS GENERATED")
    print("=" * 70)

if __name__ == "__main__":
    main()
