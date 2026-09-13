import logging
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict, Any, List
import pandas as pd
from config import DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("RiskAgent")


class RiskAgent:
    """
    Responsible for:
    - Volatility analysis (VIX level, SPY 20d annualized vol)
    - Drawdown & downside tail risk
    - Overbought/oversold extreme detection
    - Macro regime headwinds
    """
    def __init__(self):
        pass

    def evaluate_risk(self, ticker: str) -> Dict[str, Any]:
        csv_path = DATA_DIR / "training_dataset.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Training dataset not found at {csv_path}")

        df = pd.read_csv(csv_path)
        ticker_df = df[df["ticker"] == ticker].copy()
        if ticker_df.empty:
            raise ValueError(f"No records found for {ticker}")

        latest = ticker_df.sort_values(by="date").iloc[-1]

        rsi = float(latest.get("rsi", 50.0))
        vix = float(latest.get("vix_level", 18.0))
        spy_vol = float(latest.get("spy_volatility", 0.15))
        atr = float(latest.get("atr", 3.0))
        close = float(latest.get("close", 100.0)) if "close" in latest else 120.0
        atr_pct = (atr / close) if close > 0 else 0.03
        sector_ret = float(latest.get("sector_return_5d", 0.0))

        risk_factors: List[str] = []
        risk_score = 0.30  # Baseline moderate risk

        # 1. RSI Overbought Check
        if rsi >= 75:
            risk_factors.append(f"RSI extended at {rsi:.1f} (overbought exhaustion risk)")
            risk_score += 0.25
        elif rsi >= 70:
            risk_factors.append(f"RSI elevated at {rsi:.1f} (approaching overbought band)")
            risk_score += 0.15

        # 2. VIX Volatility Regime Check
        if vix >= 25:
            risk_factors.append(f"Elevated macro fear (CBOE VIX at {vix:.1f})")
            risk_score += 0.25
        elif vix >= 20:
            risk_factors.append(f"Moderate macro volatility (CBOE VIX at {vix:.1f})")
            risk_score += 0.10

        # 3. SPY Volatility Spike
        if spy_vol >= 0.20:
            risk_factors.append(f"S&P 500 rolling volatility elevated ({spy_vol:.1%})")
            risk_score += 0.15

        # 4. Single-stock ATR expansion
        if atr_pct >= 0.04:
            risk_factors.append(f"High intraday price volatility (ATR represents {atr_pct:.1%} of price)")
            risk_score += 0.10

        # 5. Sector Headwinds
        if sector_ret < -0.03:
            risk_factors.append(f"Sector-wide headwinds (5-day peer group return down {sector_ret:.1%})")
            risk_score += 0.15

        risk_score = min(1.0, max(0.10, risk_score))

        if risk_score >= 0.65:
            risk_level = "HIGH"
        elif risk_score >= 0.40:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        if not risk_factors:
            risk_factors.append("Benign macro regime and balanced momentum oscillators")

        summary = (
            f"Overall risk assessment: {risk_level} (Risk Score: {risk_score:.2f}). "
            f"Key risk considerations include: {'; '.join(risk_factors)}."
        )

        return {
            "risk_level": risk_level,
            "risk_score": round(float(risk_score), 2),
            "risk_factors": risk_factors,
            "vix_level": round(vix, 2),
            "spy_volatility": round(spy_vol, 4),
            "summary": summary,
        }


if __name__ == "__main__":
    ra = RiskAgent()
    res = ra.evaluate_risk("NVDA")
    print("\nRiskAgent Evaluation for NVDA:")
    print(f"Risk Level: {res['risk_level']} (Score: {res['risk_score']})")
    for rf in res["risk_factors"]:
        print(f" - {rf}")
