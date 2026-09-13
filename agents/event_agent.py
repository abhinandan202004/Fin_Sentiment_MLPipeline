import logging
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import List, Dict, Any
from nlp.event_detection.classifier import EventClassifier
from agents.news_agent import NewsAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("EventAgent")


class EventAgent:
    """
    Responsible for:
    - Event extraction from news headlines
    - Event classification (taxonomy mapping)
    - Impact estimation (polarity, score, and confidence)
    """
    def __init__(self):
        self.classifier = EventClassifier.get_instance()
        self.news_agent = NewsAgent()

    def detect_events_for_ticker(self, ticker: str, limit: int = 8) -> List[Dict[str, Any]]:
        articles = self.news_agent.get_recent_articles(ticker, limit=limit)
        events_found = []

        seen_events = set()
        for art in articles:
            classification = self.classifier.classify_headline(art["title"])
            ev_type = classification["event"]

            # Avoid spamming duplicates of same event type
            if ev_type in seen_events and ev_type != "market_chatter":
                continue
            seen_events.add(ev_type)

            events_found.append({
                "headline": art["title"],
                "event": ev_type,
                "impact": classification["impact"],
                "impact_score": classification["impact_score"],
                "confidence": classification["confidence"],
                "source": art["source"],
                "date": art["published_at"][:10] if art.get("published_at") else "",
            })

        # Rank events by absolute impact score and confidence
        events_found.sort(key=lambda x: (abs(x["impact_score"]), x["confidence"]), reverse=True)
        return events_found


if __name__ == "__main__":
    ea = EventAgent()
    evs = ea.detect_events_for_ticker("NVDA")
    print(f"\nEventAgent: Detected {len(evs)} events for NVDA:")
    for e in evs:
        print(f" - [{e['event'].upper()}] ({e['impact']} | {e['confidence']:.2f}): {e['headline']}")
