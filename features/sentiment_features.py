import logging
from datetime import datetime, date
from typing import Optional, List
import pandas as pd
from sqlalchemy.orm import Session
from database.connection import SessionLocal
from database.models import Article

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SentimentFeatures")


def compute_daily_sentiment_features(
    tickers: Optional[List[str]] = None,
    db: Optional[Session] = None,
) -> pd.DataFrame:
    """
    Aggregates FinBERT sentiment predictions per ticker and date:
      - avg_sentiment
      - positive_article_count
      - negative_article_count
      - total_article_count
      - sentiment_ema_3
      - sentiment_ema_7
      - sentiment_delta = today_sentiment - yesterday_sentiment
    """
    should_close = False
    if db is None:
        db = SessionLocal()
        should_close = True

    try:
        query = db.query(Article).filter(Article.sentiment_score.isnot(None))
        if tickers:
            query = query.filter(Article.ticker.in_(tickers))

        articles = query.all()
        if not articles:
            logger.warning("No scored articles found in database.")
            return pd.DataFrame(columns=[
                "ticker", "date", "avg_sentiment", "positive_article_count",
                "negative_article_count", "total_article_count",
                "sentiment_ema_3", "sentiment_ema_7", "sentiment_delta"
            ])

        rows = []
        for a in articles:
            art_date = a.published_at.date() if isinstance(a.published_at, datetime) else a.published_at
            rows.append({
                "ticker": a.ticker,
                "date": art_date,
                "sentiment_score": float(a.sentiment_score),
                "is_positive": 1 if a.sentiment_label == "positive" else 0,
                "is_negative": 1 if a.sentiment_label == "negative" else 0,
            })

        df_raw = pd.DataFrame(rows)

        # Daily aggregate
        daily = df_raw.groupby(["ticker", "date"]).agg(
            avg_sentiment=("sentiment_score", "mean"),
            positive_article_count=("is_positive", "sum"),
            negative_article_count=("is_negative", "sum"),
            total_article_count=("sentiment_score", "count"),
        ).reset_index()

        daily["date"] = pd.to_datetime(daily["date"]).dt.date

        # Compute per-ticker sentiment momentum & delta
        results = []
        for ticker, group in daily.groupby("ticker"):
            group_sorted = group.sort_values(by="date").copy()
            sent = group_sorted["avg_sentiment"]
            group_sorted["sentiment_ema_3"] = sent.ewm(span=3, adjust=False).mean()
            group_sorted["sentiment_ema_7"] = sent.ewm(span=7, adjust=False).mean()
            group_sorted["sentiment_delta"] = sent.diff(1).fillna(0.0)
            results.append(group_sorted)

        df_sentiment = pd.concat(results, ignore_index=True)
        return df_sentiment

    except Exception as e:
        logger.error(f"Error computing daily sentiment features: {e}")
        raise
    finally:
        if should_close:
            db.close()
