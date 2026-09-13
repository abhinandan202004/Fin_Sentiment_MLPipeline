import logging
from typing import List, Dict, Any, Optional
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sqlalchemy.orm import Session
from database.connection import SessionLocal
from database.models import Article
from config import FINBERT_MODEL_NAME, DEVICE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FinBERT")


class FinBERTSentimentAnalyzer:
    _instance: Optional["FinBERTSentimentAnalyzer"] = None

    def __init__(self, model_name: str = FINBERT_MODEL_NAME, device: str = DEVICE):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and device != "cpu" else "cpu"
        )
        logger.info(f"Loading FinBERT model '{model_name}' on device '{self.device}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # FinBERT labels mapping
        self.labels = [self.model.config.id2label[i].lower() for i in range(len(self.model.config.id2label))]
        logger.info(f"FinBERT loaded successfully with classes: {self.labels}")

    @classmethod
    def get_instance(cls) -> "FinBERTSentimentAnalyzer":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def predict(self, texts: List[str], max_length: int = 512) -> List[Dict[str, Any]]:
        """
        Runs batched FinBERT inference.
        Returns for each text:
          - 'label': 'positive', 'negative', or 'neutral'
          - 'score': compound score = prob(positive) - prob(negative) in [-1.0, 1.0]
          - 'probs': dict of {label: prob}
        """
        if not texts:
            return []

        cleaned_texts = [t.strip() if t and t.strip() else "Neutral" for t in texts]
        
        inputs = self.tokenizer(
            cleaned_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = F.softmax(outputs.logits, dim=-1).cpu().numpy()

        results = []
        for probs in probabilities:
            prob_dict = {self.labels[i]: float(probs[i]) for i in range(len(self.labels))}
            pos = prob_dict.get("positive", 0.0)
            neg = prob_dict.get("negative", 0.0)
            neu = prob_dict.get("neutral", 0.0)

            # Compound score in [-1.0, 1.0]
            compound_score = float(pos - neg)
            predicted_label = max(prob_dict, key=prob_dict.get)

            results.append({
                "label": predicted_label,
                "score": compound_score,
                "probs": prob_dict,
            })

        return results


def score_unscored_articles(batch_size: int = 16, db: Session = None) -> int:
    """
    Finds all articles in the database missing sentiment scores and updates them.
    """
    should_close_db = False
    if db is None:
        db = SessionLocal()
        should_close_db = True

    try:
        analyzer = FinBERTSentimentAnalyzer.get_instance()
        unscored = db.query(Article).filter(Article.sentiment_score.is_(None)).all()

        if not unscored:
            logger.info("No unscored articles found.")
            return 0

        logger.info(f"Found {len(unscored)} articles to score with FinBERT...")
        total_scored = 0

        for i in range(0, len(unscored), batch_size):
            batch = unscored[i:i + batch_size]
            texts = [
                f"{a.title}. {a.content or ''}"[:1000]
                for a in batch
            ]
            predictions = analyzer.predict(texts)

            for article, pred in zip(batch, predictions):
                article.sentiment_score = pred["score"]
                article.sentiment_label = pred["label"]
                article.sentiment_probs = pred["probs"]
                total_scored += 1

            db.commit()
            logger.info(f"Scored {total_scored}/{len(unscored)} articles...")

        return total_scored

    except Exception as e:
        db.rollback()
        logger.error(f"Error scoring articles: {e}")
        raise
    finally:
        if should_close_db:
            db.close()
