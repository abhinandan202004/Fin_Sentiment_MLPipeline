import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import logging
import re
from typing import Dict, Any, List, Optional
from nlp.event_detection.taxonomy import EVENT_TYPES, EVENT_WEIGHTS, EVENT_PATTERNS
from nlp.sentiment import FinBERTSentimentAnalyzer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("EventClassifier")


class EventClassifier:
    _instance: Optional["EventClassifier"] = None

    def __init__(self):
        self.analyzer = FinBERTSentimentAnalyzer.get_instance()
        # Compile patterns
        self.compiled_patterns = {
            event: [re.compile(p, re.IGNORECASE) for p in patterns]
            for event, patterns in EVENT_PATTERNS.items()
        }

    @classmethod
    def get_instance(cls) -> "EventClassifier":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def classify_headline(self, headline: str) -> Dict[str, Any]:
        """
        Classifies financial headline into a standard financial event type,
        determining impact ('positive', 'negative', 'neutral'), impact_score, and confidence.
        """
        if not headline or not headline.strip():
            return {
                "event": "market_chatter",
                "impact": "neutral",
                "impact_score": 0.0,
                "confidence": 0.50,
            }

        text = headline.strip()
        matched_events = []

        # 1. Check taxonomy regex patterns
        for event, patterns in self.compiled_patterns.items():
            for pat in patterns:
                match = pat.search(text)
                if match:
                    base_weight = EVENT_WEIGHTS.get(event, 0.0)
                    matched_events.append((event, base_weight, 0.85))
                    break

        # 2. FinBERT sentiment verification
        finbert_res = self.analyzer.predict([text])[0]
        sentiment_score = finbert_res["score"]
        sentiment_label = finbert_res["label"]

        if matched_events:
            # Pick strongest matched event
            matched_events.sort(key=lambda x: abs(x[1]), reverse=True)
            top_event, base_weight, pat_conf = matched_events[0]

            # Adjust impact_score with FinBERT sentiment alignment
            combined_score = 0.65 * base_weight + 0.35 * sentiment_score
            combined_score = max(-1.0, min(1.0, combined_score))

            if combined_score > 0.15:
                impact = "positive"
            elif combined_score < -0.15:
                impact = "negative"
            else:
                impact = "neutral"

            confidence = min(0.98, max(0.60, pat_conf + 0.10 * abs(sentiment_score)))

            return {
                "event": top_event,
                "impact": impact,
                "impact_score": round(float(combined_score), 2),
                "confidence": round(float(confidence), 2),
            }

        # 3. Fallback if no specific template matched: classify by FinBERT sentiment
        if sentiment_label == "positive":
            event = "analyst_upgrade" if sentiment_score > 0.6 else "product_launch"
            impact = "positive"
        elif sentiment_label == "negative":
            event = "supply_chain_issue" if sentiment_score < -0.6 else "regulation"
            impact = "negative"
        else:
            event = "routine_disclosure"
            impact = "neutral"

        conf = min(0.85, max(0.50, abs(sentiment_score)))
        return {
            "event": event,
            "impact": impact,
            "impact_score": round(float(sentiment_score), 2),
            "confidence": round(float(conf), 2),
        }

    def classify_batch(self, headlines: List[str]) -> List[Dict[str, Any]]:
        return [self.classify_headline(h) for h in headlines]


if __name__ == "__main__":
    classifier = EventClassifier.get_instance()
    test_headlines = [
        "NVIDIA beats earnings expectations and raises guidance",
        "Apple faces supply chain bottlenecks and shipment delays in Q3",
        "Wall Street upgrades NVDA to Strong Buy citing Blackwell GPU demand",
        "Federal Reserve hikes interest rates by 25 basis points",
    ]
    for h in test_headlines:
        res = classifier.classify_headline(h)
        print(f"\nHeadline: '{h}'")
        print(f"Result: {res}")
