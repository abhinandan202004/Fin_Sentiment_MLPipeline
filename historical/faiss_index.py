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
    "rsi",
    "macd",
    "volume_ratio",
    "event_impact",
    "spy_volatility",
    "spy_return_5d",
]


class AnalogIndexBuilder:
    def __init__(self):
        self.scaler = StandardScaler()
        self.index: Optional[faiss.IndexFlatL2] = None
        self.metadata: Dict[int, Dict[str, Any]] = {}

    def build_index_from_dataset(self, csv_path: Optional[Path] = None) -> Tuple[faiss.IndexFlatL2, Dict[int, Dict[str, Any]]]:
        if csv_path is None:
            csv_path = DATA_DIR / "training_dataset.csv"

        if not csv_path.exists():
            raise FileNotFoundError(f"Training dataset not found at {csv_path}")

        logger.info(f"Loading observations from {csv_path} for FAISS analog index...")
        df = pd.read_csv(csv_path)

        # Approximate event_impact from sentiment & return spikes if not explicit
        if "event_impact" not in df.columns:
            # Map sentiment + volume breakout into event impact proxy
            df["event_impact"] = np.clip(df["avg_sentiment"] * 0.7 + np.sign(df["return_1d"]) * 0.3, -1.0, 1.0)

        # Filter valid rows
        cols_to_check = [c for c in ANALOG_FEATURE_COLS if c in df.columns]
        clean_df = df.dropna(subset=cols_to_check + ["future_return_5d"]).copy().reset_index(drop=True)

        features_raw = clean_df[ANALOG_FEATURE_COLS].values.astype(np.float32)
        scaled_features = self.scaler.fit_transform(features_raw)

        # Build FAISS IndexFlatL2
        dim = scaled_features.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(scaled_features)

        # Build Metadata dictionary
        self.metadata = {}
        for idx, row in clean_df.iterrows():
            spy_vol = row.get("spy_volatility", 0.15)
            if spy_vol < 0.14:
                regime = "low_vol"
            elif spy_vol > 0.22:
                regime = "high_vol"
            elif row.get("spy_return_5d", 0) > 0.01:
                regime = "bull_trend"
            else:
                regime = "bear_trend"

            ev_label = "market_catalyst" if abs(row.get("event_impact", 0)) > 0.4 else "neutral_regime"
            if row.get("event_impact", 0) > 0.4:
                ev_label = "earnings_beat_or_upgrade"
            elif row.get("event_impact", 0) < -0.4:
                ev_label = "earnings_miss_or_headwind"

            analog_case = AnalogCase(
                vector_id=int(idx),
                date=str(row.get("date", "")),
                ticker=str(row.get("ticker", "")),
                event=ev_label,
                return_1d=round(float(row.get("future_return_1d", 0.0)), 4),
                return_5d=round(float(row.get("future_return_5d", 0.0)), 4),
                return_10d=round(float(row.get("future_return_10d", 0.0)), 4),
                sentiment=round(float(row.get("avg_sentiment", 0.0)), 4),
                rsi=round(float(row.get("rsi", 50.0)), 2),
                volume_ratio=round(float(row.get("volume_ratio", 1.0)), 2),
                market_regime=regime,
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
    idx, meta = builder.build_index_from_dataset()
    print(f"FAISS Index Total Vectors: {idx.ntotal}")
    print(f"Sample Metadata (Vector 0): {meta[0]}")
