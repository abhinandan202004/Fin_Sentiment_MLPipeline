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
from database.models import TrainingFeature, MarketData, Article, Prediction
from ingestion.market.yfinance_ingestor import YFinanceIngestor
from ingestion.news import fetch_and_store_news
from nlp.sentiment import score_unscored_articles
from features.technical import compute_ticker_technical_indicators, fetch_market_regime
from features.sentiment_features import compute_daily_sentiment_features

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

        # Initialize SHAP explainer
        self.explainer = shap.TreeExplainer(self.model)

    def get_latest_feature_vector(self, ticker: str) -> Tuple[pd.Series, str]:
        """
        Retrieves the latest available feature vector for ticker from database
        or live yfinance computation.
        """
        db = SessionLocal()
        try:
            # 1. Check feature store table first
            latest_tf = (
                db.query(TrainingFeature)
                .filter(TrainingFeature.ticker == ticker)
                .order_by(TrainingFeature.date.desc())
                .first()
            )

            if latest_tf:
                row_dict = {f: getattr(latest_tf, f, 0.0) for f in self.features}
                # Clean nulls
                for k, v in row_dict.items():
                    if v is None:
                        row_dict[k] = 0.0
                return pd.Series(row_dict), str(latest_tf.date)

            # 2. Fallback: live computation via yfinance
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
        Returns standardized JSON contract for future Agent layer.
        """
        feature_vector, as_of_date = self.get_latest_feature_vector(ticker)
        X = pd.DataFrame([feature_vector[self.features]])

        # Inference
        pred_label_int = int(self.model.predict(X)[0])
        probabilities = self.model.predict_proba(X)[0]
        confidence = float(probabilities[1]) if pred_label_int == 1 else float(probabilities[0])
        prediction_str = "BUY" if pred_label_int == 1 else "NOT BUY"

        # Expected 5-day return proxy (scaled by probability edge above 50%)
        prob_buy = float(probabilities[1])
        base_expected_ret = (prob_buy - 0.50) * 0.10  # ~2-4% expected move based on conviction

        # Volatility & Risk calculation
        vol = float(feature_vector.get("atr", 3.0)) / (float(feature_vector.get("close", 100.0)) + 1e-5)
        if vol > 0.04:
            risk = "HIGH"
        elif vol > 0.02:
            risk = "MEDIUM"
        else:
            risk = "LOW"

        # SHAP Instance Explanation
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

        top_pos = [f"{d['feature']} ({d['impact']:+.3f})" for d in positive_drivers[:3]]
        top_neg = [f"{d['feature']} ({d['impact']:+.3f})" for d in negative_drivers[:3]]

        result = {
            "ticker": ticker.upper(),
            "date": as_of_date,
            "prediction": prediction_str,
            "confidence": round(confidence, 4),
            "expected_5d_return": round(base_expected_ret, 4),
            "risk": risk,
            "top_positive_drivers": top_pos,
            "top_negative_drivers": top_neg,
            "model_version": self.metadata.get("model", "optuna_tuned_v1"),
        }

        # Persist to database predictions table
        db = SessionLocal()
        try:
            db_pred = Prediction(
                ticker=ticker.upper(),
                prediction=prediction_str,
                confidence=confidence,
                model_version=self.metadata.get("model", "optuna_tuned_v1"),
                shap_drivers={"positive": positive_drivers, "negative": negative_drivers},
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
