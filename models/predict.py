import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import logging
from datetime import datetime, date, timezone
from typing import Dict, Any, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
import shap

from config import ARTIFACTS_DIR, DATA_DIR
from database.connection import SessionLocal
from database.models import TrainingFeature, MarketData, Prediction

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("PredictionService")


class PredictionService:
    def __init__(self):
        self.model_path = ARTIFACTS_DIR / "best_model.pkl"
        self.feature_path = ARTIFACTS_DIR / "feature_list.json"
        self.meta_path = ARTIFACTS_DIR / "metadata.json"

        if not self.model_path.exists() or not self.feature_path.exists():
            raise FileNotFoundError(
                f"Trained model artifacts not found at {ARTIFACTS_DIR}. Please run models/train.py first."
            )

        self.model = joblib.load(self.model_path)
        with open(self.feature_path, "r") as f:
            self.features = json.load(f)

        self.metadata = {}
        if self.meta_path.exists():
            with open(self.meta_path, "r") as f:
                self.metadata = json.load(f)

        self.threshold_path = ARTIFACTS_DIR / "optimal_threshold.json"
        self.threshold = 0.55
        if self.threshold_path.exists():
            try:
                with open(self.threshold_path, "r") as f:
                    th_info = json.load(f)
                    self.threshold = float(th_info.get("optimal_threshold", 0.55))
            except Exception:
                self.threshold = 0.55

        # Initialize SHAP explainer
        self.explainer = shap.TreeExplainer(self.model)

    def get_latest_feature_vector(self, ticker: str) -> Tuple[pd.Series, str]:
        """
        Retrieves the latest available feature vector for ticker.
        Prioritizes the 30,025-row engineered dataset (43 features across 25 tickers).
        Falls back to database / yfinance if ticker is outside the core universe.
        """
        ticker = ticker.upper()
        try:
            from models.feature_store import load_feature_dataset
            df = load_feature_dataset()
            ticker_df = df[df["ticker"] == ticker].sort_values(by="date")
            if not ticker_df.empty:
                latest_row = ticker_df.iloc[-1]
                row_dict = {}
                for f in self.features:
                    val = latest_row.get(f, 0.0)
                    row_dict[f] = float(val) if pd.notna(val) else 0.0
                return pd.Series(row_dict), str(latest_row.get("date", date.today()))
        except Exception as e:
            logger.warning(f"Feature store retrieval fallback for {ticker}: {e}")

        # Fallback to DB or live computation
        db = SessionLocal()
        try:
            latest_tf = (
                db.query(TrainingFeature)
                .filter(TrainingFeature.ticker == ticker)
                .order_by(TrainingFeature.date.desc())
                .first()
            )

            if latest_tf:
                row_dict = {f: getattr(latest_tf, f, 0.0) for f in self.features}
                for k, v in row_dict.items():
                    if v is None:
                        row_dict[k] = 0.0
                return pd.Series(row_dict), str(latest_tf.date)

            # Live computation via yfinance
            logger.info(f"Building live feature vector for {ticker}...")
            ingestor = YFinanceIngestor(db=db)
            ingestor.ingest_ticker(ticker, period="6mo", db=db)
            bars = (
                db.query(MarketData)
                .filter(MarketData.ticker == ticker)
                .order_by(MarketData.date.asc())
                .all()
            )
            df_mkt = pd.DataFrame([{"date": b.date, "open": b.open, "high": b.high, "low": b.low, "close": b.close, "volume": b.volume} for b in bars])
            tech_df = compute_ticker_technical_indicators(df_mkt)
            latest_tech = tech_df.iloc[-1]

            row_dict = {}
            for f in self.features:
                if f in latest_tech:
                    row_dict[f] = float(latest_tech[f]) if pd.notna(latest_tech[f]) else 0.0
                else:
                    row_dict[f] = 0.0

            return pd.Series(row_dict), str(latest_tech["Date"].date() if "Date" in latest_tech else date.today())
        finally:
            db.close()

    def predict(self, ticker: str) -> Dict[str, Any]:
        """
        Executes end-to-end prediction for ticker.
        Outputs calibrated probability, optimal threshold, decision signal,
        and fine-grained TreeSHAP feature attributions.
        """
        feature_vector, as_of_date = self.get_latest_feature_vector(ticker)
        X = pd.DataFrame([feature_vector[self.features]])

        # Model Probabilities & Calibrated Threshold Decision
        probabilities = self.model.predict_proba(X)[0]
        prob_buy = float(probabilities[1])
        prediction_str = "BUY" if prob_buy >= self.threshold else "NOT BUY"
        confidence = prob_buy if prediction_str == "BUY" else float(probabilities[0])

        # Expected 5-day return proxy
        base_expected_ret = (prob_buy - self.threshold) * 0.10

        # Volatility & Risk calculation
        vix = float(feature_vector.get("vix_level", 20.0))
        vol = float(feature_vector.get("atr", 3.0)) / (float(feature_vector.get("close", 100.0)) + 1e-5)
        if vix > 25.0 or vol > 0.04:
            risk = "HIGH"
        elif vix < 16.0 and vol < 0.02:
            risk = "LOW"
        else:
            risk = "MEDIUM"

        # Market Regime identification
        spy_5d = float(feature_vector.get("spy_return_5d", 0.0))
        spy_vol = float(feature_vector.get("spy_volatility", 0.15))
        if vix > 25.0 or spy_vol > 0.22:
            market_regime = "High Volatility"
        elif vix < 15.0 and spy_vol < 0.14:
            market_regime = "Low Volatility"
        elif spy_5d > 0.01:
            market_regime = "Bull"
        elif spy_5d < -0.01:
            market_regime = "Bear"
        else:
            market_regime = "Neutral"

        # SHAP Tree Explanations
        shap_vals = self.explainer.shap_values(X)
        if isinstance(shap_vals, list):
            sv = shap_vals[1][0] if len(shap_vals) > 1 else shap_vals[0][0]
        elif len(shap_vals.shape) == 3:
            sv = shap_vals[0, :, 1]
        else:
            sv = shap_vals[0]

        drivers = []
        for feat, val, s_val in zip(self.features, X.iloc[0], sv):
            drivers.append({
                "feature": feat,
                "value": round(float(val), 4),
                "impact": round(float(s_val), 4),
            })

        positive_drivers = sorted([d for d in drivers if d["impact"] > 0], key=lambda x: x["impact"], reverse=True)
        negative_drivers = sorted([d for d in drivers if d["impact"] < 0], key=lambda x: x["impact"])

        top_pos = [f"{d['feature']} ({d['impact']:+.3f})" for d in positive_drivers[:4]]
        top_neg = [f"{d['feature']} ({d['impact']:+.3f})" for d in negative_drivers[:4]]

        quant_features = {
            "sentiment_rank": round(float(feature_vector.get("sentiment_rank", 0.5)), 3),
            "vix_level": round(vix, 2),
            "market_regime": market_regime,
            "bb_squeeze": round(float(feature_vector.get("bb_squeeze", 0.0)), 2),
            "sector_return_5d": round(float(feature_vector.get("sector_return_5d", 0.0)), 4),
            "rsi_rank": round(float(feature_vector.get("rsi_rank", 0.5)), 3),
            "volume_surge": int(feature_vector.get("volume_surge", 0)),
        }

        result = {
            "ticker": ticker.upper(),
            "date": as_of_date,
            "signal": prediction_str,
            "prediction": prediction_str,
            "probability": round(prob_buy, 4),
            "threshold": round(self.threshold, 2),
            "confidence": round(confidence, 4),
            "expected_5d_return": round(base_expected_ret, 4),
            "risk": risk,
            "market_regime": market_regime,
            "top_positive_drivers": top_pos,
            "top_negative_drivers": top_neg,
            "feature_contributions": drivers,
            "quant_features": quant_features,
            "model_version": self.metadata.get("model", "Random Forest"),
        }

        # Persist to database predictions table
        db = SessionLocal()
        try:
            db_pred = Prediction(
                ticker=ticker.upper(),
                prediction=prediction_str,
                confidence=confidence,
                model_version=self.metadata.get("model", "Random Forest"),
                shap_drivers={"positive": positive_drivers[:5], "negative": negative_drivers[:5]},
            )
            db.add(db_pred)
            db.commit()
        except Exception as e:
            db.rollback()
            logger.debug(f"Failed to persist prediction record: {e}")
        finally:
            db.close()

        return result


def predict_ticker(ticker: str) -> Dict[str, Any]:
    """Convenience function for python callers."""
    service = PredictionService()
    return service.predict(ticker)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Stock Prediction Service")
    parser.add_argument("--ticker", type=str, default="NVDA", help="Ticker symbol (e.g. NVDA)")
    args = parser.parse_args()

    svc = PredictionService()
    res = svc.predict(args.ticker)
    print("\n" + json.dumps(res, indent=2) + "\n")
