import logging
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from database.connection import SessionLocal
from database.models import Article
from ingestion.news import fetch_and_store_news

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("NewsAgent")


class NewsAgent:
    """
    Responsible for:
    - News collection & live retrieval
    - Ranking articles by relevance and recency
    - Source credibility scoring
    """
    def __init__(self):
        pass

    def get_recent_articles(
        self,
        ticker: str,
        limit: int = 10,
        refresh_live: bool = False,
    ) -> List[Dict[str, Any]]:
        if refresh_live:
            try:
                fetch_and_store_news(ticker)
            except Exception as e:
                logger.warning(f"Could not refresh live news for {ticker}: {e}")

        db: Session = SessionLocal()
        try:
            articles = (
                db.query(Article)
                .filter(Article.ticker == ticker)
                .order_by(Article.published_at.desc())
                .limit(limit * 2)
                .all()
            )

            results = []
            for a in articles:
                # Calculate source credibility weight
                source = a.source or "Unknown"
                cred = a.credibility_score or 1.0
                if "Reuters" in source or "Bloomberg" in source or "Financial Times" in source:
                    cred = max(cred, 0.95)
                elif "Yahoo" in source or "PR Newswire" in source:
                    cred = max(cred, 0.85)

                results.append({
                    "id": a.id,
                    "ticker": a.ticker,
                    "title": a.title,
                    "source": source,
                    "published_at": a.published_at.isoformat() if a.published_at else "",
                    "sentiment_score": float(a.sentiment_score) if a.sentiment_score is not None else 0.0,
                    "sentiment_label": a.sentiment_label or "neutral",
                    "credibility_score": cred,
                })

            # Sort by credibility * recency rank
            results.sort(key=lambda x: (x["credibility_score"], x["published_at"]), reverse=True)
            return results[:limit]

        finally:
            db.close()


if __name__ == "__main__":
    agent = NewsAgent()
    arts = agent.get_recent_articles("NVDA", limit=5)
    print(f"\nNewsAgent: Retrieved {len(arts)} articles for NVDA:")
    for a in arts:
        print(f" - [{a['source']} | Cred={a['credibility_score']:.2f}] {a['title']}")
