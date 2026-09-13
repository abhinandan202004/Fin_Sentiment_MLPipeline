import logging
from datetime import datetime, timezone
import hashlib
from typing import List, Dict, Any
import requests
import feedparser
import yfinance as yf
from sqlalchemy.orm import Session
from database.connection import SessionLocal
from database.models import Article

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NewsIngestion")


def parse_timestamp(val: Any) -> datetime:
    """Parses various timestamp representations (unix int, string) to UTC datetime."""
    if isinstance(val, (int, float)):
        return datetime.fromtimestamp(val, tz=timezone.utc).replace(tzinfo=None)
    elif isinstance(val, str):
        try:
            return datetime.fromisoformat(val.replace("Z", "+00:00")).replace(tzinfo=None)
        except Exception:
            pass
    return datetime.now(timezone.utc).replace(tzinfo=None)


def fetch_yfinance_news(ticker: str) -> List[Dict[str, Any]]:
    """Retrieves current articles for a ticker via yfinance."""
    articles = []
    try:
        stock = yf.Ticker(ticker)
        raw_news = stock.news or []
        for item in raw_news:
            title = item.get("title")
            if not title:
                continue

            # yfinance news schema can vary: content can be under 'summary', 'body', or providerPublishTime
            summary = item.get("summary") or item.get("content") or ""
            link = item.get("link") or item.get("url") or ""
            source = item.get("publisher") or item.get("provider") or "Yahoo Finance"
            pub_time = item.get("providerPublishTime") or item.get("pubDate")
            published_at = parse_timestamp(pub_time)

            articles.append({
                "ticker": ticker,
                "source": source,
                "title": title.strip(),
                "content": summary.strip() if summary else title.strip(),
                "url": link.strip(),
                "published_at": published_at,
            })
    except Exception as e:
        logger.warning(f"Failed to fetch yfinance news for {ticker}: {e}")
    return articles


def fetch_rss_news(ticker: str) -> List[Dict[str, Any]]:
    """Retrieves articles via Yahoo Finance RSS feed."""
    articles = []
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    try:
        feed = feedparser.parse(url)
        for entry in feed.entries:
            title = entry.get("title")
            if not title:
                continue

            link = entry.get("link") or ""
            summary = entry.get("summary") or entry.get("description") or ""
            published_parsed = entry.get("published_parsed")
            if published_parsed:
                published_at = datetime(*published_parsed[:6])
            else:
                published_at = datetime.now(timezone.utc).replace(tzinfo=None)

            articles.append({
                "ticker": ticker,
                "source": "Yahoo RSS",
                "title": title.strip(),
                "content": summary.strip() if summary else title.strip(),
                "url": link.strip(),
                "published_at": published_at,
            })
    except Exception as e:
        logger.warning(f"Failed to fetch RSS news for {ticker}: {e}")
    return articles


def fetch_and_store_news(
    ticker: str,
    db: Session = None,
) -> int:
    """
    Fetches news from Yahoo Finance APIs & RSS feeds, deduplicates against
    existing database records, and stores raw articles.
    """
    should_close_db = False
    if db is None:
        db = SessionLocal()
        should_close_db = True

    try:
        logger.info(f"Fetching news for {ticker}...")
        yf_news = fetch_yfinance_news(ticker)
        rss_news = fetch_rss_news(ticker)
        all_candidates = yf_news + rss_news

        new_count = 0
        for item in all_candidates:
            # Check duplicate by URL or unique Title
            url = item["url"]
            title = item["title"]

            query = db.query(Article)
            if url:
                existing = query.filter((Article.url == url) | (Article.title == title)).first()
            else:
                existing = query.filter(Article.title == title).first()

            if existing:
                continue

            article = Article(
                ticker=item["ticker"],
                source=item["source"],
                title=item["title"],
                content=item["content"],
                url=item["url"] if item["url"] else None,
                published_at=item["published_at"],
                credibility_score=1.0,
            )
            db.add(article)
            new_count += 1

        db.commit()
        logger.info(f"Stored {new_count} new news articles for {ticker}.")
        return new_count

    except Exception as e:
        db.rollback()
        logger.error(f"Error storing news for {ticker}: {e}")
        raise
    finally:
        if should_close_db:
            db.close()
