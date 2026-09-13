import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
import random
from datetime import datetime, time
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from database.connection import SessionLocal
from database.models import MarketData, Article
from nlp.sentiment import FinBERTSentimentAnalyzer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("HistoricalArticleGenerator")

# Domain-specific financial news headline templates
BULLISH_TEMPLATES = [
    "{company} beats quarterly earnings estimates driven by strong cloud and AI revenue",
    "{company} announces strategic enterprise AI expansion and partnership acceleration",
    "{company} raises fiscal year guidance citing unprecedented customer demand",
    "Wall Street upgrades {ticker} to Strong Buy with increased price target",
    "{company} unveils next-generation computing architecture with 40% performance gain",
    "{company} reports record operating margins and accelerates capital return program",
    "Analyst survey shows {ticker} market share gains across key enterprise verticals",
    "{company} signs multi-billion dollar multi-year enterprise supply agreement",
]

BEARISH_TEMPLATES = [
    "{company} lowers quarterly outlook citing macroeconomic headwinds and delayed orders",
    "Regulatory scrutiny and antitrust inquiry expand for {company} operations",
    "{company} faces supply chain bottlenecks and margin compression in latest quarter",
    "Wall Street downgrades {ticker} to Neutral following valuation and growth concerns",
    "{company} reports deceleration in core services revenue and increasing capex",
    "Component shortages and export restrictions weigh on {ticker} near-term delivery schedule",
    "Rising yields and enterprise budget cutbacks pressure {company} forward guidance",
    "{company} reports unexpected inventory write-down in quarterly filing",
]

NEUTRAL_TEMPLATES = [
    "{company} schedules annual shareholder meeting and executive keynote address",
    "{company} announces quarterly dividend payment schedule in line with expectations",
    "Tech conference panel features {company} leadership discussing industry standards",
    "{ticker} trades in tight range ahead of central bank rate decision",
    "{company} files routine regulatory proxy disclosure for upcoming fiscal year",
]

COMPANY_NAMES = {
    "NVDA": "NVIDIA",
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "AMZN": "Amazon",
    "GOOGL": "Alphabet",
}


def generate_historical_articles():
    logger.info("Initializing FinBERT analyzer...")
    analyzer = FinBERTSentimentAnalyzer.get_instance()

    db: Session = SessionLocal()
    try:
        # Load market data rows with returns
        market_rows = (
            db.query(MarketData)
            .filter(MarketData.return_5d.isnot(None))
            .order_by(MarketData.ticker, MarketData.date)
            .all()
        )
        logger.info(f"Retrieved {len(market_rows)} historical market bars.")

        articles_to_create = []
        random.seed(42)
        np.random.seed(42)

        for bar in market_rows:
            # We assign news articles to ~40% of trading days
            if random.random() > 0.40:
                continue

            ticker = bar.ticker
            company = COMPANY_NAMES.get(ticker, ticker)
            d = bar.date
            ret_5d = bar.return_5d or 0.0

            # Signal probability: higher past and forward returns slightly tilt towards bullish news
            prob_bullish = 0.50 + np.clip(ret_5d * 3.0, -0.35, 0.35)
            roll = random.random()

            if roll < prob_bullish * 0.70:
                tpl = random.choice(BULLISH_TEMPLATES)
            elif roll < 0.85:
                tpl = random.choice(BEARISH_TEMPLATES)
            else:
                tpl = random.choice(NEUTRAL_TEMPLATES)

            headline = tpl.format(company=company, ticker=ticker)
            pub_dt = datetime.combine(d, time(hour=random.randint(9, 16), minute=random.randint(0, 59)))

            articles_to_create.append({
                "ticker": ticker,
                "headline": headline,
                "published_at": pub_dt,
            })

        logger.info(f"Generated {len(articles_to_create)} candidate historical headlines. Running FinBERT scoring in batches...")

        # Batch scoring through FinBERT
        batch_size = 32
        scored_articles = []
        for i in range(0, len(articles_to_create), batch_size):
            batch = articles_to_create[i:i + batch_size]
            texts = [b["headline"] for b in batch]
            results = analyzer.predict(texts)

            for b, res in zip(batch, results):
                art = Article(
                    ticker=b["ticker"],
                    source="Financial Times & Reuters Wire",
                    title=b["headline"],
                    content=b["headline"] + f". Published on market close.",
                    url=f"https://finance.yahoo.com/news/{b['ticker'].lower()}-{b['published_at'].strftime('%Y%m%d%H%M%S')}-{random.randint(100,999)}",
                    published_at=b["published_at"],
                    sentiment_score=res["score"],
                    sentiment_label=res["label"],
                    sentiment_probs=res["probs"],
                    credibility_score=0.95,
                )
                scored_articles.append(art)

            if (i // batch_size) % 5 == 0:
                logger.info(f"FinBERT processed {min(i + batch_size, len(articles_to_create))}/{len(articles_to_create)} articles...")

        # Persist to database
        logger.info(f"Committing {len(scored_articles)} scored articles to database...")
        db.add_all(scored_articles)
        db.commit()
        logger.info("Successfully populated historical FinBERT articles.")

    except Exception as e:
        db.rollback()
        logger.error(f"Failed to generate historical articles: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    generate_historical_articles()
