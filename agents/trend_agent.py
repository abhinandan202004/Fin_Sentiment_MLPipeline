"""
Trend Agent for Sprint 8: Autonomous Research & Market Opportunity Discovery.

Identifies, tracks, and ranks macro & secular market themes:
  - Artificial Intelligence & Machine Learning
  - Semiconductors & Advanced Computing
  - Cloud Infrastructure & Enterprise Software
  - Cybersecurity & Defense Intelligence
  - Defense & Aerospace
  - Energy Transition & Oil/Gas
  - Healthcare, Pharma & Biotechnology
  - FinTech & Digital Banking

Measures for each theme:
  - Mention growth: % change in article mentions over 7 days vs baseline
  - Sentiment trend: FinBERT polarity shift (bullish/bearish momentum)
  - News volume: Total thematic article count
  - Sector strength: Aggregated market price momentum of peer assets
  - Leading tickers: Top direct beneficiary assets in the coverage universe

Persists theme snapshots to the `trend_history` table.

Usage:
    python agents/trend_agent.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone, timedelta

from database.connection import SessionLocal, init_db
from database.models import TrendHistory, Article, TrainingFeature

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("TrendAgent")


THEMES_DEFINITIONS = {
    "AI & Machine Learning": {
        "keywords": ["ai", "artificial intelligence", "machine learning", "deep learning", "llm", "generative ai", "copilot", "chatgpt"],
        "beneficiaries": ["NVDA", "MSFT", "GOOGL", "META", "AMD", "ORCL"]
    },
    "Semiconductors": {
        "keywords": ["chip", "chips", "semiconductor", "foundry", "wafer", "gpu", "tsmc", "nanometer", "memory", "intel", "amd"],
        "beneficiaries": ["NVDA", "AMD", "INTC"]
    },
    "Cloud & Enterprise Software": {
        "keywords": ["cloud", "saas", "azure", "aws", "datacenter", "enterprise software", "hyperscaler", "database"],
        "beneficiaries": ["MSFT", "AMZN", "GOOGL", "ORCL", "CRM"]
    },
    "Cybersecurity": {
        "keywords": ["cybersecurity", "ransomware", "breach", "zero trust", "firewall", "endpoint", "security threat"],
        "beneficiaries": ["MSFT", "CRWD", "PANW", "GOOGL"]
    },
    "Defense & Aerospace": {
        "keywords": ["defense", "military", "aerospace", "pentagon", "missile", "drone", "nato", "defense contract"],
        "beneficiaries": ["HON", "LMT", "RTX", "BA"]
    },
    "Energy & Oil/Gas": {
        "keywords": ["oil", "gas", "energy", "crude", "opec", "refinery", "pipeline", "fossil fuel", "drilling"],
        "beneficiaries": ["XOM", "CVX"]
    },
    "Healthcare & Biotech": {
        "keywords": ["drug", "fda", "clinical trial", "pharma", "biotech", "vaccine", "glp-1", "obesity drug", "oncology"],
        "beneficiaries": ["LLY", "PFE", "JNJ", "UNH"]
    },
    "FinTech & Capital Markets": {
        "keywords": ["banking", "fed", "interest rate", "fintech", "payments", "credit card", "investment banking", "trading"],
        "beneficiaries": ["JPM", "GS", "BAC", "V"]
    }
}


class TrendAgent:
    """Detects and monitors high-growth macro and sector themes."""

    def __init__(self):
        init_db()

    def analyze_themes(self, days_recent: int = 7, days_baseline: int = 30) -> List[Dict[str, Any]]:
        """
        Analyzes news article volume and sentiment shifts across all secular themes.
        Computes mention growth rate, sentiment trend, and maps top beneficiary tickers.
        """
        session = SessionLocal()
        now = datetime.now(timezone.utc)
        recent_cutoff = now - timedelta(days=days_recent)
        baseline_cutoff = now - timedelta(days=days_baseline)

        try:
            # Query recent and baseline articles
            recent_articles = (
                session.query(Article)
                .filter(Article.published_at >= recent_cutoff)
                .all()
            )
            baseline_articles = (
                session.query(Article)
                .filter(Article.published_at >= baseline_cutoff)
                .all()
            )

            # Query 5d sector return from features for sector strength
            sector_strength_map = {}
            features = session.query(TrainingFeature).order_by(TrainingFeature.date.desc()).limit(25).all()
            for f in features:
                if f.ticker and f.sector_return_5d is not None:
                    sector_strength_map[f.ticker] = f.sector_return_5d

            theme_results = []

            for theme_name, theme_info in THEMES_DEFINITIONS.items():
                kw_list = [k.lower() for k in theme_info["keywords"]]
                beneficiaries = theme_info["beneficiaries"]

                # Count recent mentions
                recent_matches = [
                    a for a in recent_articles
                    if any(kw in (a.title or "").lower() or kw in (a.content or "").lower() for kw in kw_list)
                ]
                recent_count = len(recent_matches)

                # Count baseline mentions (normalized to weekly equivalent)
                baseline_matches = [
                    a for a in baseline_articles
                    if any(kw in (a.title or "").lower() or kw in (a.content or "").lower() for kw in kw_list)
                ]
                baseline_weekly = max(1.0, len(baseline_matches) * (days_recent / days_baseline))

                # Growth rate
                mention_growth = (recent_count - baseline_weekly) / baseline_weekly

                # Sentiment trend
                sentiments = [a.sentiment_score for a in recent_matches if a.sentiment_score is not None]
                avg_sent = float(sum(sentiments) / len(sentiments)) if sentiments else 0.0

                # Sector strength
                sec_returns = [sector_strength_map.get(t, 0.0) for t in beneficiaries if t in sector_strength_map]
                avg_strength = float(sum(sec_returns) / len(sec_returns)) if sec_returns else 0.0

                theme_results.append({
                    "theme": theme_name,
                    "mention_growth": round(mention_growth, 4),
                    "sentiment_trend": round(avg_sent, 4),
                    "news_volume": recent_count,
                    "sector_strength": round(avg_strength, 4),
                    "leading_tickers": beneficiaries
                })

            # Sort descending by mention_growth
            theme_results.sort(key=lambda x: (x["mention_growth"], x["news_volume"]), reverse=True)
            return theme_results

        finally:
            session.close()

    def snapshot_trends(self, save_to_db: bool = True) -> List[Dict[str, Any]]:
        """Analyzes themes and saves snapshots to trend_history."""
        themes = self.analyze_themes()
        if not save_to_db:
            return themes

        session = SessionLocal()
        today = datetime.now(timezone.utc).date()
        try:
            for t in themes:
                rec = TrendHistory(
                    detected_date=today,
                    theme=t["theme"],
                    mention_growth=t["mention_growth"],
                    sentiment_trend=t["sentiment_trend"],
                    news_volume=t["news_volume"],
                    sector_strength=t["sector_strength"],
                    leading_tickers=t["leading_tickers"]
                )
                session.add(rec)
            session.commit()
            logger.info(f"Persisted {len(themes)} trend snapshots to trend_history.")
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to persist trend history: {e}")
        finally:
            session.close()

        return themes

    def get_theme_for_ticker(self, ticker: str) -> Optional[str]:
        """Returns the primary thematic alignment for a ticker."""
        ticker = ticker.upper().strip()
        for theme_name, theme_info in THEMES_DEFINITIONS.items():
            if ticker in theme_info["beneficiaries"]:
                return theme_name
        return None


if __name__ == "__main__":
    agent = TrendAgent()
    trends = agent.snapshot_trends(save_to_db=True)
    print("\n" + "=" * 75)
    print("Sprint 8: Macro & Thematic Trend Intelligence")
    print("=" * 75)
    print(f"{'Theme':<30} {'Growth':<12} {'Sentiment':<12} {'Articles':<10} {'Leading Assets'}")
    print("-" * 75)
    for t in trends:
        growth_str = f"{t['mention_growth']:+.1%}"
        sent_str = f"{t['sentiment_trend']:+.2f}"
        tickers_str = ", ".join(t['leading_tickers'][:3])
        print(f"{t['theme']:<30} {growth_str:<12} {sent_str:<12} {t['news_volume']:<10} {tickers_str}")
    print("=" * 75)
