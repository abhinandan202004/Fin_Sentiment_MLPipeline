import logging
from datetime import datetime, date, timedelta
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session
from database.connection import SessionLocal
from database.models import MarketData, Article, Event

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FeatureBuilder")

FEATURE_COLUMNS = [
    "sentiment_score",
    "article_count",
    "pos_article_count",
    "neg_article_count",
    "volume_change",
    "rsi",
    "macd",
    "volatility",
    "sentiment_momentum",
    "event_impact_score",
]

TARGET_COLUMN = "target_buy"


def build_feature_dataset(
    tickers: Optional[List[str]] = None,
    db: Session = None,
) -> pd.DataFrame:
    """
    Constructs the research feature matrix aligning daily market technicals,
    FinBERT news sentiment metrics, and event scores.
    """
    should_close_db = False
    if db is None:
        db = SessionLocal()
        should_close_db = True

    try:
        # 1. Fetch Market Data
        market_query = db.query(MarketData)
        if tickers:
            market_query = market_query.filter(MarketData.ticker.in_(tickers))
        market_records = market_query.order_by(MarketData.ticker, MarketData.date).all()

        if not market_records:
            logger.warning("No market data found in database.")
            return pd.DataFrame()

        mkt_rows = [
            {
                "ticker": r.ticker,
                "date": r.date,
                "close": r.close,
                "volume": r.volume,
                "rsi": r.rsi,
                "macd": r.macd,
                "macd_signal": r.macd_signal,
                "volatility": r.volatility,
                "return_1d": r.return_1d,
                "return_5d": r.return_5d,
                "return_10d": r.return_10d,
            }
            for r in market_records
        ]
        df_market = pd.DataFrame(mkt_rows)
        df_market["date"] = pd.to_datetime(df_market["date"]).dt.date

        # Compute rolling volume change: (Vol - SMA20(Vol)) / SMA20(Vol) per ticker
        def calc_vol_change_series(s):
            sma = s.rolling(20, min_periods=5).mean()
            return (s - sma) / (sma + 1e-5)

        df_market["volume_change"] = df_market.groupby("ticker")["volume"].transform(calc_vol_change_series)

        # 2. Fetch News Articles & Aggregate Daily Sentiment
        article_query = db.query(Article).filter(Article.sentiment_score.isnot(None))
        if tickers:
            article_query = article_query.filter(Article.ticker.in_(tickers))
        articles = article_query.all()

        if articles:
            art_rows = []
            for a in articles:
                art_date = a.published_at.date()
                art_rows.append({
                    "ticker": a.ticker,
                    "date": art_date,
                    "sentiment_score": a.sentiment_score,
                    "is_positive": 1 if a.sentiment_label == "positive" else 0,
                    "is_negative": 1 if a.sentiment_label == "negative" else 0,
                })
            df_art = pd.DataFrame(art_rows)

            daily_sentiment = df_art.groupby(["ticker", "date"]).agg(
                sentiment_score=("sentiment_score", "mean"),
                article_count=("sentiment_score", "count"),
                pos_article_count=("is_positive", "sum"),
                neg_article_count=("is_negative", "sum"),
            ).reset_index()
        else:
            daily_sentiment = pd.DataFrame(columns=["ticker", "date", "sentiment_score", "article_count", "pos_article_count", "neg_article_count"])

        # 3. Fetch Events & Aggregate Impact Scores
        event_query = db.query(Event)
        if tickers:
            event_query = event_query.filter(Event.ticker.in_(tickers))
        events = event_query.all()

        if events:
            evt_rows = [
                {
                    "ticker": e.ticker,
                    "date": e.detected_at.date(),
                    "event_impact_score": e.impact_score,
                }
                for e in events
            ]
            df_evt = pd.DataFrame(evt_rows)
            daily_events = df_evt.groupby(["ticker", "date"]).agg(
                event_impact_score=("event_impact_score", "sum")
            ).reset_index()
        else:
            daily_events = pd.DataFrame(columns=["ticker", "date", "event_impact_score"])

        # 4. Merge Market + Sentiment + Events
        merged = pd.merge(df_market, daily_sentiment, on=["ticker", "date"], how="left")
        merged = pd.merge(merged, daily_events, on=["ticker", "date"], how="left")

        # Fill sentiment & event missing days
        merged["sentiment_score"] = merged["sentiment_score"].fillna(0.0).astype(float)
        merged["article_count"] = merged["article_count"].fillna(0).astype(int)
        merged["pos_article_count"] = merged["pos_article_count"].fillna(0).astype(int)
        merged["neg_article_count"] = merged["neg_article_count"].fillna(0).astype(int)
        merged["event_impact_score"] = merged["event_impact_score"].fillna(0.0).astype(float)

        # Vectorized sentiment momentum (3-day EMA - 7-day EMA)
        ema_fast = merged.groupby("ticker")["sentiment_score"].transform(lambda s: s.ewm(span=3, adjust=False).mean())
        ema_slow = merged.groupby("ticker")["sentiment_score"].transform(lambda s: s.ewm(span=7, adjust=False).mean())
        merged["sentiment_momentum"] = (ema_fast - ema_slow).astype(float)

        # Ensure all feature columns are strictly numeric float/int
        for col in ["volume_change", "rsi", "macd", "volatility"]:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

        # 5. Define Supervised Binary Target: 1 if return_5d > 0 else 0
        merged[TARGET_COLUMN] = (merged["return_5d"] > 0).astype(int)

        # Filter out trailing rows where return_5d is NaN (the latest 5 trading days)
        valid_dataset = merged.dropna(subset=["return_5d", "rsi", "macd", "volatility", "volume_change"]).copy()

        logger.info(f"Engineered feature dataset: {len(valid_dataset)} rows across {valid_dataset['ticker'].nunique()} tickers.")
        return valid_dataset

    except Exception as e:
        logger.error(f"Error building feature dataset: {e}")
        raise
    finally:
        if should_close_db:
            db.close()
