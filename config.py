import os
from pathlib import Path
from dotenv import load_dotenv

# Base paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "models" / "artifacts"

DATA_DIR.mkdir(exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Load environment variables
load_dotenv(BASE_DIR / ".env")

# Database URLs
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/fin_sentiment_db"
)
SQLITE_FALLBACK_URL = os.getenv(
    "SQLITE_FALLBACK_URL",
    f"sqlite:///{DATA_DIR / 'fin_sentiment.db'}"
)

# Ingestion settings
DEFAULT_TICKERS = [
    t.strip().upper()
    for t in os.getenv("DEFAULT_TICKERS", "NVDA,AAPL,MSFT,AMZN,GOOGL").split(",")
    if t.strip()
]
MARKET_HISTORY_PERIOD = os.getenv("MARKET_HISTORY_PERIOD", "2y")

# NLP settings
FINBERT_MODEL_NAME = os.getenv("FINBERT_MODEL_NAME", "ProsusAI/finbert")
DEVICE = os.getenv("DEVICE", "cpu")

# ML Settings
TARGET_HORIZON_DAYS = int(os.getenv("TARGET_HORIZON_DAYS", "5"))
TRAIN_TEST_SPLIT_RATIO = float(os.getenv("TRAIN_TEST_SPLIT_RATIO", "0.8"))
XGB_N_ESTIMATORS = int(os.getenv("XGB_N_ESTIMATORS", "100"))
XGB_LEARNING_RATE = float(os.getenv("XGB_LEARNING_RATE", "0.05"))
XGB_MAX_DEPTH = int(os.getenv("XGB_MAX_DEPTH", "4"))
