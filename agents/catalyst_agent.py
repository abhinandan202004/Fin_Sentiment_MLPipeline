"""
Catalyst Agent for Sprint 8: Autonomous Research & Market Opportunity Discovery.

Identifies and evaluates corporate and market catalysts:
  - Earnings & Forward Guidance
  - Product Launches & Technical Keynotes
  - Acquisitions, Mergers & Divestitures
  - Regulatory, Legal & Antitrust Actions
  - Analyst Upgrades, Downgrades & Price Targets
  - Supply Chain & Strategic Partnerships

Ranks catalysts by signed expected impact on asset valuation in [-1.0, +1.0]
and persists findings to the `catalyst_history` table.

Usage:
    python agents/catalyst_agent.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import re
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone, timedelta

from database.connection import SessionLocal, init_db
from database.models import CatalystHistory, Article, Event

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("CatalystAgent")


CATALYST_PATTERNS = {
    "Earnings & Guidance": {
        "keywords": ["earnings", "guidance", "q1", "q2", "q3", "q4", "revenue beat", "eps beat", "profit", "quarterly result", "revenue miss"],
        "base_impact": 0.40
    },
    "Product Launches": {
        "keywords": ["launch", "unveil", "keynote", "announces new", "release", "rollout", "next-gen", "architecture", "flagship"],
        "base_impact": 0.35
    },
    "Acquisitions & M&A": {
        "keywords": ["acquire", "acquisition", "merger", "takeover", "buyout", "deal", "divestiture", "purchase"],
        "base_impact": 0.30
    },
    "Regulation & Legal": {
        "keywords": ["antitrust", "sec", "ftc", "doj", "lawsuit", "regulatory", "probe", "investigation", "fine", "compliance"],
        "base_impact": -0.35
    },
    "Analyst Upgrades": {
        "keywords": ["upgrade", "downgrade", "price target", "outperform", "overweight", "underweight", "analyst buy", "top pick"],
        "base_impact": 0.25
    },
    "Supply Chain & Partnerships": {
        "keywords": ["partnership", "collaboration", "supplier", "supply chain", "contract", "agreement", "foundry deal"],
        "base_impact": 0.20
    }
}


class CatalystAgent:
    """Detects, classifies, and scores the impact of market and corporate catalysts."""

    def __init__(self):
        init_db()

    def classify_text(self, text: str, sentiment_score: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        Scans a headline or event description to identify catalyst type and impact.
        Returns catalyst metadata dict or None if no catalyst pattern is detected.
        """
        if not text:
            return None

        text_lower = text.lower()
        matched_category = None
        matched_keywords = []

        for category, config in CATALYST_PATTERNS.items():
            hits = [kw for kw in config["keywords"] if re.search(r'\b' + re.escape(kw) + r'\b', text_lower)]
            if hits:
                matched_category = category
                matched_keywords = hits
                break

        if not matched_category:
            return None

        base_impact = CATALYST_PATTERNS[matched_category]["base_impact"]

        # Adjust impact using FinBERT sentiment score if available
        if sentiment_score is not None:
            # sentiment_score ranges -1.0 to +1.0
            sentiment_adjust = sentiment_score * 0.40
            # If negative headline in an upgrade or earnings category, flip sign appropriately
            if "miss" in text_lower or "downgrade" in text_lower or "drop" in text_lower:
                expected_impact = -abs(base_impact) - abs(sentiment_adjust)
            else:
                expected_impact = (base_impact if base_impact >= 0 else -abs(base_impact)) + sentiment_adjust
        else:
            if "miss" in text_lower or "downgrade" in text_lower or "drop" in text_lower:
                expected_impact = -abs(base_impact)
            else:
                expected_impact = base_impact

        expected_impact = max(-1.0, min(1.0, round(expected_impact, 2)))

        return {
            "catalyst_type": matched_category,
            "matched_keywords": matched_keywords,
            "expected_impact": expected_impact,
            "description": text.strip()
        }

    def detect_catalysts_for_ticker(self, ticker: str, days_back: int = 14) -> List[Dict[str, Any]]:
        """Scans recent articles and events for a specific ticker to extract catalysts."""
        ticker = ticker.upper().strip()
        session = SessionLocal()
        cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
        catalysts = []

        try:
            # Check articles
            articles = (
                session.query(Article)
                .filter(Article.ticker == ticker, Article.published_at >= cutoff)
                .order_by(Article.published_at.desc())
                .limit(20)
                .all()
            )
            for art in articles:
                classified = self.classify_text(art.title, art.sentiment_score)
                if classified:
                    classified.update({
                        "ticker": ticker,
                        "detected_date": art.published_at.date() if art.published_at else datetime.now(timezone.utc).date(),
                        "source_url": art.url
                    })
                    catalysts.append(classified)

            # Check corporate events table
            events = (
                session.query(Event)
                .filter(Event.ticker == ticker, Event.detected_at >= cutoff)
                .order_by(Event.detected_at.desc())
                .limit(10)
                .all()
            )
            for ev in events:
                classified = self.classify_text(ev.description, ev.impact_score)
                if classified:
                    classified.update({
                        "ticker": ticker,
                        "detected_date": ev.detected_at.date() if ev.detected_at else datetime.now(timezone.utc).date(),
                        "source_url": None
                    })
                    catalysts.append(classified)

            # Sort descending by absolute impact
            catalysts.sort(key=lambda x: abs(x["expected_impact"]), reverse=True)
            return catalysts

        finally:
            session.close()

    def scan_all_catalysts(self, days_back: int = 14, save_to_db: bool = True) -> List[Dict[str, Any]]:
        """
        Scans all articles and events across the entire coverage universe and
        persists detected catalysts to the catalyst_history table.
        """
        session = SessionLocal()
        cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
        today = datetime.now(timezone.utc).date()
        detected = []

        try:
            articles = (
                session.query(Article)
                .filter(Article.published_at >= cutoff, Article.ticker.isnot(None))
                .order_by(Article.published_at.desc())
                .all()
            )

            seen_descriptions = set()

            for art in articles:
                if not art.title or art.title in seen_descriptions:
                    continue
                seen_descriptions.add(art.title)

                c = self.classify_text(art.title, art.sentiment_score)
                if c:
                    c.update({
                        "ticker": art.ticker,
                        "detected_date": art.published_at.date() if art.published_at else today,
                        "source_url": art.url
                    })
                    detected.append(c)

            detected.sort(key=lambda x: abs(x["expected_impact"]), reverse=True)

            if save_to_db and detected:
                for cat in detected[:50]:  # Top 50 catalysts
                    rec = CatalystHistory(
                        detected_date=cat["detected_date"],
                        ticker=cat["ticker"],
                        catalyst_type=cat["catalyst_type"],
                        description=cat["description"],
                        expected_impact=cat["expected_impact"],
                        source_url=cat.get("source_url")
                    )
                    session.add(rec)
                session.commit()
                logger.info(f"Persisted {len(detected[:50])} catalysts to catalyst_history.")

            return detected

        except Exception as e:
            session.rollback()
            logger.error(f"Failed to scan and persist catalysts: {e}")
            return detected
        finally:
            session.close()


if __name__ == "__main__":
    agent = CatalystAgent()
    catalysts = agent.scan_all_catalysts(days_back=14, save_to_db=True)
    print("\n" + "=" * 80)
    print("Sprint 8: Corporate & Market Catalyst Intelligence")
    print(f"Total Catalysts Detected: {len(catalysts)}")
    print("=" * 80)
    print(f"{'Ticker':<8} {'Impact':<10} {'Catalyst Category':<30} {'Headline / Trigger'}")
    print("-" * 80)
    for c in catalysts[:10]:
        imp_str = f"{c['expected_impact']:+.2f}"
        desc = (c['description'][:40] + "...") if len(c['description']) > 40 else c['description']
        print(f"{c['ticker']:<8} {imp_str:<10} {c['catalyst_type']:<30} {desc}")
    print("=" * 80)
