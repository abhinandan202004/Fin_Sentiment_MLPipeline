import logging
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
import faiss
import joblib
from sklearn.preprocessing import StandardScaler

from config import DATA_DIR, ARTIFACTS_DIR
from historical.analog_metadata import AnalogCase

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("FAISSIndex")

INDEX_PATH = ARTIFACTS_DIR / "faiss_analogs.index"
METADATA_PATH = ARTIFACTS_DIR / "faiss_metadata.json"
SCALER_PATH = ARTIFACTS_DIR / "faiss_scaler.pkl"

ANALOG_FEATURE_COLS = [
    "avg_sentiment",
    "sentiment_rank",
    "rsi",
    "macd",
    "volume_ratio",
    "event_impact",
    "spy_volatility",
    "vix_level",
    "sector_return_5d",
    "bb_squeeze",
]

TICKER_SECTOR_MAP = {
    "NVDA": "Technology",
    "AAPL": "Technology",
    "MSFT": "Technology",
    "GOOGL": "Communication Services",
    "META": "Communication Services",
    "AMZN": "Consumer Discretionary",
    "TSLA": "Consumer Discretionary",
    "NFLX": "Communication Services",
    "AMD": "Technology",
    "INTC": "Technology",
    "CRM": "Technology",
    "ORCL": "Technology",
    "JPM": "Financials",
    "BAC": "Financials",
    "GS": "Financials",
    "V": "Financials",
    "UNH": "Healthcare",
    "JNJ": "Healthcare",
    "PFE": "Healthcare",
    "XOM": "Energy",
    "CVX": "Energy",
    "WMT": "Consumer Staples",
    "HD": "Consumer Discretionary",
    "CAT": "Industrials",
    "HON": "Industrials",
}


class AnalogIndexBuilder:
    def __init__(self):
        self.scaler = StandardScaler()
        self.index: Optional[faiss.IndexFlatL2] = None
        self.metadata: Dict[int, Dict[str, Any]] = {}

    def build_index_from_dataset(self, csv_path: Optional[Path] = None) -> Tuple[faiss.IndexFlatL2, Dict[int, Dict[str, Any]], StandardScaler]:
        from models.feature_store import load_feature_dataset

        logger.info("Loading engineered feature dataset (30k rows, 25 tickers) for FAISS analog index...")
        if csv_path is not None and csv_path.exists():
            df = pd.read_csv(csv_path)
            if "sentiment_rank" not in df.columns:
                from models.feature_engineer import engineer_features
                df = engineer_features(df)
        else:
            df = load_feature_dataset(run_feature_engineering=True)

        # Approximate event_impact from sentiment & return spikes if not explicit
        if "event_impact" not in df.columns:
            ret_col = "return_1d" if "return_1d" in df.columns else "future_return_1d"
            df["event_impact"] = np.clip(df["avg_sentiment"] * 0.7 + np.sign(df[ret_col]) * 0.3, -1.0, 1.0)

        # Ensure all analog feature columns exist and are clean
        for col in ANALOG_FEATURE_COLS:
            if col not in df.columns:
                df[col] = 0.0

        # Filter valid rows with clean future returns
        clean_df = df.dropna(subset=ANALOG_FEATURE_COLS + ["future_return_5d"]).copy().reset_index(drop=True)

        features_raw = clean_df[ANALOG_FEATURE_COLS].values.astype(np.float32)
        scaled_features = self.scaler.fit_transform(features_raw)

        # Build FAISS IndexFlatL2
        dim = scaled_features.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(scaled_features)

        # Build Metadata dictionary
        self.metadata = {}
        for idx, row in clean_df.iterrows():
            vix = float(row.get("vix_level", 20.0))
            spy_vol = float(row.get("spy_volatility", 0.15))
            spy_5d = float(row.get("spy_return_5d", 0.0))

            if vix > 25.0 or spy_vol > 0.22:
                regime = "High Volatility"
            elif vix < 15.0 and spy_vol < 0.14:
                regime = "Low Volatility"
            elif spy_5d > 0.01:
                regime = "Bull"
            elif spy_5d < -0.01:
                regime = "Bear"
            else:
                regime = "Neutral"

            ev_score = float(row.get("event_impact", 0.0))
            if ev_score > 0.4:
                ev_label = "earnings_beat_or_upgrade"
            elif ev_score < -0.4:
                ev_label = "earnings_miss_or_headwind"
            elif abs(ev_score) > 0.2:
                ev_label = "market_catalyst"
            else:
                ev_label = "neutral_regime"

            t = str(row.get("ticker", ""))
            sec = TICKER_SECTOR_MAP.get(t, "Other")

            analog_case = AnalogCase(
                vector_id=int(idx),
                date=str(row.get("date", "")),
                ticker=t,
                event=ev_label,
                return_1d=round(float(row.get("future_return_1d", 0.0)), 4),
                return_5d=round(float(row.get("future_return_5d", 0.0)), 4),
                return_10d=round(float(row.get("future_return_10d", 0.0)), 4),
                sentiment=round(float(row.get("avg_sentiment", 0.0)), 4),
                rsi=round(float(row.get("rsi", 50.0)), 2),
                volume_ratio=round(float(row.get("volume_ratio", 1.0)), 2),
                market_regime=regime,
                sector=sec,
                vix_level=round(vix, 2),
                sentiment_rank=round(float(row.get("sentiment_rank", 0.5)), 3),
                sector_return_5d=round(float(row.get("sector_return_5d", 0.0)), 4),
                bb_squeeze=round(float(row.get("bb_squeeze", 0.0)), 2),
            )
            self.metadata[int(idx)] = analog_case.to_dict()

        logger.info(f"FAISS index populated with {self.index.ntotal} historical vector embeddings (dim={dim}).")
        self.save_artifacts()
        return self.index, self.metadata, self.scaler

    def save_artifacts(self):
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        if self.index is not None:
            faiss.write_index(self.index, str(INDEX_PATH))
        with open(METADATA_PATH, "w") as f:
            json.dump(self.metadata, f, indent=2)
        joblib.dump(self.scaler, SCALER_PATH)
        logger.info(f"FAISS index and metadata successfully persisted to {ARTIFACTS_DIR}")

    def load_artifacts(self) -> Tuple[faiss.IndexFlatL2, Dict[int, Dict[str, Any]], StandardScaler]:
        if not INDEX_PATH.exists() or not METADATA_PATH.exists() or not SCALER_PATH.exists():
            logger.info("Existing FAISS artifacts not found. Building index from dataset...")
            return self.build_index_from_dataset()

        self.index = faiss.read_index(str(INDEX_PATH))
        with open(METADATA_PATH, "r") as f:
            raw_meta = json.load(f)
            self.metadata = {int(k): v for k, v in raw_meta.items()}
        self.scaler = joblib.load(SCALER_PATH)
        logger.info(f"Loaded existing FAISS index ({self.index.ntotal} vectors) from {ARTIFACTS_DIR}")
        return self.index, self.metadata, self.scaler


if __name__ == "__main__":
    builder = AnalogIndexBuilder()
    idx, meta, scaler = builder.build_index_from_dataset()
    print(f"FAISS Index Total Vectors: {idx.ntotal}")
    print(f"Sample Metadata (Vector 0): {meta[0]}")
