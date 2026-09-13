import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import logging
from datetime import datetime, date
import pandas as pd
from sqlalchemy.orm import Session

from config import DEFAULT_TICKERS, MARKET_HISTORY_PERIOD, DATA_DIR
from database.connection import init_db, SessionLocal
from database.models import TrainingFeature, MarketData
from ingestion.market.yfinance_ingestor import YFinanceIngestor
from ingestion.news import fetch_and_store_news
from nlp.sentiment import score_unscored_articles
from features.technical import compute_all_technical_features, fetch_market_regime, fetch_spy_regime
from features.sentiment_features import compute_daily_sentiment_features
from notebooks.eda import run_eda
from models.baseline_model import train_logistic_regression_baseline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("DatasetBuilder")

TRAINING_COLUMNS = [
    "ticker",
    "date",
    "avg_sentiment",
    "sentiment_ema_3",
    "sentiment_ema_7",
    "sentiment_delta",
    "positive_article_count",
    "negative_article_count",
    "total_article_count",
    "rsi",
    "macd",
    "macd_signal",
    "macd_diff",
    "ema20",
    "ema50",
    "bollinger_high",
    "bollinger_low",
    "bollinger_pband",
    "atr",
    "return_1d",
    "return_5d",
    "return_20d",
    "volume_ratio",
    "spy_return_1d",
    "spy_return_5d",
    "spy_volatility",
    "vix_level",
    "sector_return_5d",
    "future_return_1d",
    "future_return_5d",
    "future_return_10d",
    "target",
]


def build_training_dataset(
    tickers: list = None,
    period: str = MARKET_HISTORY_PERIOD,
    skip_news: bool = False,
) -> pd.DataFrame:
    if not tickers:
        tickers = DEFAULT_TICKERS

    print("\n" + "=" * 80)
    print(" SPRINT 2: PREDICTIVE DATASET & FEATURE STORE PIPELINE")
    print(f" Target Tickers : {', '.join(tickers)}")
    print(f" History Period : {period}")
    print("=" * 80 + "\n")

    # Step 1: Ensure database schema
    logger.info("Step 1/10: Initializing database schema...")
    init_db()

    # Step 2: Refresh Market Data (Tickers + SPY Benchmark)
    logger.info("Step 2/10: Refreshing market data with yfinance...")
    ingestor = YFinanceIngestor()
    for t in tickers:
        ingestor.ingest_ticker(t, period=period)

    # Step 3: Refresh Financial News
    if not skip_news:
        logger.info("Step 3/10: Refreshing news headlines...")
        for t in tickers:
            try:
                fetch_and_store_news(t)
            except Exception as e:
                logger.warning(f"News fetch skipped for {t}: {e}")
    else:
        logger.info("Step 3/10: Skipping live news fetch (--skip-news enabled).")

    # Step 4: Run FinBERT Scoring
    logger.info("Step 4/10: Running FinBERT NLP sentiment inference on unscored articles...")
    scored = score_unscored_articles()
    logger.info(f"FinBERT scored {scored} new articles.")

    # Step 5: Load Market Data from DB
    logger.info("Step 5/10: Loading market price bars from database...")
    db = SessionLocal()
    try:
        market_rows = (
            db.query(MarketData)
            .filter(MarketData.ticker.in_(tickers))
            .order_by(MarketData.ticker, MarketData.date)
            .all()
        )
        if not market_rows:
            raise ValueError("No market data bars found in database.")

        df_market = pd.DataFrame([
            {
                "ticker": r.ticker,
                "date": r.date,
                "open": r.open,
                "high": r.high,
                "low": r.low,
                "close": r.close,
                "volume": r.volume,
            }
            for r in market_rows
        ])
    finally:
        db.close()

    # Step 6: Compute Technical Features & Market Regime
    logger.info("Step 6/10: Computing technical indicators (RSI, MACD, EMAs, BB, ATR, SPY regime, VIX)...")
    regime_df = fetch_market_regime(period=period)
    df_technical = compute_all_technical_features(df_market, regime_df=regime_df)

    # Step 7: Compute Daily Sentiment Features
    logger.info("Step 7/10: Aggregating daily sentiment features (counts, EMA3, EMA7, delta)...")
    df_sentiment = compute_daily_sentiment_features(tickers=tickers)

    # Step 8: Merge Technicals + Sentiment into Training Dataset
    logger.info("Step 8/10: Assembling master feature store...")
    if not df_sentiment.empty:
        df_master = pd.merge(df_technical, df_sentiment, on=["ticker", "date"], how="left")
    else:
        df_master = df_technical.copy()
        for col in ["avg_sentiment", "sentiment_ema_3", "sentiment_ema_7", "sentiment_delta",
                    "positive_article_count", "negative_article_count", "total_article_count"]:
            df_master[col] = 0.0

    # Fill default values for days without articles
    df_master["avg_sentiment"] = df_master["avg_sentiment"].fillna(0.0).astype(float)
    df_master["sentiment_ema_3"] = df_master["sentiment_ema_3"].fillna(0.0).astype(float)
    df_master["sentiment_ema_7"] = df_master["sentiment_ema_7"].fillna(0.0).astype(float)
    df_master["sentiment_delta"] = df_master["sentiment_delta"].fillna(0.0).astype(float)
    df_master["positive_article_count"] = df_master["positive_article_count"].fillna(0).astype(int)
    df_master["negative_article_count"] = df_master["negative_article_count"].fillna(0).astype(int)
    df_master["total_article_count"] = df_master["total_article_count"].fillna(0).astype(int)

    # Ensure all technical columns are valid numbers
    tech_cols = ["rsi", "macd", "macd_signal", "macd_diff", "ema20", "ema50",
                 "bollinger_high", "bollinger_low", "bollinger_pband", "atr",
                 "return_1d", "return_5d", "return_20d", "volume_ratio",
                 "spy_return_1d", "spy_return_5d", "spy_volatility", "vix_level", "sector_return_5d"]
    for c in tech_cols:
        df_master[c] = pd.to_numeric(df_master[c], errors="coerce")

    # Drop warm-up rows (first 50 days where EMA50/RSI/ATR are NaN) and trailing rows without 5d future return
    clean_df = df_master.dropna(subset=tech_cols + ["future_return_5d"]).copy()
    clean_df["target"] = clean_df["target"].astype(int)

    # Select ordered columns
    available_cols = [c for c in TRAINING_COLUMNS if c in clean_df.columns]
    final_df = clean_df[available_cols].copy()

    # Step 9: Run EDA & Data Quality Verification
    logger.info("Step 9/10: Executing Data Quality Verification & EDA...")
    eda_summary = run_eda(final_df)

    # Quality Assertions
    assert not eda_summary["has_duplicates"], "Duplicate (ticker, date) rows detected!"
    assert eda_summary["target_all_valid"], "Target contains null values!"
    assert len(final_df) >= 100, f"Insufficient samples ({len(final_df)}) in training dataset!"

    # Export to CSV deliverable
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = DATA_DIR / "training_dataset.csv"
    final_df.to_csv(csv_path, index=False)
    logger.info(f"Exported training dataset ({len(final_df)} rows) to {csv_path}")

    # Step 10: Upsert into training_features Database Table
    logger.info("Step 10/10: Upserting records into training_features table...")
    db = SessionLocal()
    try:
        upsert_count = 0
        for _, row in final_df.iterrows():
            t_date = row["date"]
            t_ticker = row["ticker"]

            existing = (
                db.query(TrainingFeature)
                .filter(TrainingFeature.ticker == t_ticker, TrainingFeature.date == t_date)
                .first()
            )

            if not existing:
                tf_entry = TrainingFeature(
                    ticker=t_ticker,
                    date=t_date,
                    avg_sentiment=float(row["avg_sentiment"]),
                    sentiment_ema_3=float(row["sentiment_ema_3"]),
                    sentiment_ema_7=float(row["sentiment_ema_7"]),
                    sentiment_delta=float(row["sentiment_delta"]),
                    positive_article_count=int(row["positive_article_count"]),
                    negative_article_count=int(row["negative_article_count"]),
                    total_article_count=int(row["total_article_count"]),
                    rsi=float(row["rsi"]) if pd.notna(row["rsi"]) else None,
                    macd=float(row["macd"]) if pd.notna(row["macd"]) else None,
                    macd_signal=float(row["macd_signal"]) if pd.notna(row["macd_signal"]) else None,
                    macd_diff=float(row["macd_diff"]) if pd.notna(row["macd_diff"]) else None,
                    ema20=float(row["ema20"]) if pd.notna(row["ema20"]) else None,
                    ema50=float(row["ema50"]) if pd.notna(row["ema50"]) else None,
                    bollinger_high=float(row["bollinger_high"]) if pd.notna(row["bollinger_high"]) else None,
                    bollinger_low=float(row["bollinger_low"]) if pd.notna(row["bollinger_low"]) else None,
                    bollinger_pband=float(row["bollinger_pband"]) if pd.notna(row["bollinger_pband"]) else None,
                    atr=float(row["atr"]) if pd.notna(row["atr"]) else None,
                    return_1d=float(row["return_1d"]) if pd.notna(row["return_1d"]) else None,
                    return_5d=float(row["return_5d"]) if pd.notna(row["return_5d"]) else None,
                    return_20d=float(row["return_20d"]) if pd.notna(row["return_20d"]) else None,
                    volume_ratio=float(row["volume_ratio"]) if pd.notna(row["volume_ratio"]) else None,
                    spy_return_1d=float(row["spy_return_1d"]) if pd.notna(row.get("spy_return_1d")) else None,
                    spy_return_5d=float(row["spy_return_5d"]) if pd.notna(row["spy_return_5d"]) else None,
                    spy_volatility=float(row["spy_volatility"]) if pd.notna(row.get("spy_volatility")) else None,
                    vix_level=float(row["vix_level"]) if pd.notna(row.get("vix_level")) else None,
                    sector_return_5d=float(row["sector_return_5d"]) if pd.notna(row.get("sector_return_5d")) else None,
                    future_return_1d=float(row["future_return_1d"]) if pd.notna(row["future_return_1d"]) else None,
                    future_return_5d=float(row["future_return_5d"]) if pd.notna(row["future_return_5d"]) else None,
                    future_return_10d=float(row["future_return_10d"]) if pd.notna(row["future_return_10d"]) else None,
                    target=int(row["target"]),
                )
                db.add(tf_entry)
                upsert_count += 1

        db.commit()
        logger.info(f"Successfully populated {upsert_count} new records into training_features table.")
    except Exception as e:
        db.rollback()
        logger.error(f"Error persisting to training_features: {e}")
    finally:
        db.close()

    # Step 11: Train & Evaluate Baseline Logistic Regression
    logger.info("Training Logistic Regression Baseline benchmark...")
    clf, metrics, test_data = train_logistic_regression_baseline(final_df)

    return final_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sprint 2 Predictive Dataset & Feature Store Builder")
    parser.add_argument(
        "--tickers",
        type=str,
        default="NVDA,AAPL,MSFT,AMZN,GOOGL",
        help="Comma-separated ticker list (default: NVDA,AAPL,MSFT,AMZN,GOOGL)",
    )
    parser.add_argument(
        "--period",
        type=str,
        default="2y",
        help="Market history period (default: 2y)",
    )
    parser.add_argument(
        "--skip-news",
        action="store_true",
        help="Skip fetching live news and use cached articles",
    )
    args = parser.parse_args()

    ticker_list = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    build_training_dataset(
        tickers=ticker_list,
        period=args.period,
        skip_news=args.skip_news,
    )
