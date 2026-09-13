import uuid
from datetime import datetime, timezone
from sqlalchemy import (
    Column,
    String,
    Text,
    Float,
    BigInteger,
    DateTime,
    Date,
    JSON,
    UniqueConstraint,
    Index,
)
from .connection import Base


def generate_uuid():
    return str(uuid.uuid4())


class Company(Base):
    __tablename__ = "companies"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    ticker = Column(String(20), unique=True, nullable=False, index=True)
    company_name = Column(String(255), nullable=False)
    sector = Column(String(100), nullable=True)
    industry = Column(String(100), nullable=True)
    market_cap = Column(BigInteger, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    def __repr__(self):
        return f"<Company {self.ticker} - {self.company_name}>"


class Article(Base):
    __tablename__ = "articles"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    ticker = Column(String(20), nullable=True, index=True)
    source = Column(String(100), nullable=False)
    title = Column(Text, nullable=False)
    content = Column(Text, nullable=True)
    url = Column(Text, unique=True, nullable=True)
    published_at = Column(DateTime, nullable=False, index=True)
    sentiment_score = Column(Float, nullable=True)
    sentiment_label = Column(String(20), nullable=True)
    sentiment_probs = Column(JSON, nullable=True)  # {"positive": 0.81, "negative": 0.09, "neutral": 0.10}
    credibility_score = Column(Float, default=1.0)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("idx_article_ticker_published", "ticker", "published_at"),
    )

    def __repr__(self):
        return f"<Article {self.title[:30]} ({self.ticker})>"


class Event(Base):
    __tablename__ = "events"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    ticker = Column(String(20), nullable=True, index=True)
    event_type = Column(String(100), nullable=False)
    description = Column(Text, nullable=False)
    impact_score = Column(Float, default=0.0)
    detected_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    def __repr__(self):
        return f"<Event {self.event_type} ({self.ticker})>"


class MarketData(Base):
    __tablename__ = "market_data"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    ticker = Column(String(20), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(BigInteger, nullable=False)
    
    # Technical Indicators
    rsi = Column(Float, nullable=True)
    macd = Column(Float, nullable=True)
    macd_signal = Column(Float, nullable=True)
    volatility = Column(Float, nullable=True)
    
    # Forward Returns for Supervised Target Generation
    return_1d = Column(Float, nullable=True)
    return_5d = Column(Float, nullable=True)
    return_10d = Column(Float, nullable=True)

    __table_args__ = (
        UniqueConstraint("ticker", "date", name="uq_ticker_date"),
        Index("idx_market_ticker_date", "ticker", "date"),
    )

    def __repr__(self):
        return f"<MarketData {self.ticker} {self.date} Close={self.close}>"


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    ticker = Column(String(20), nullable=False, index=True)
    prediction = Column(String(20), nullable=False)  # BUY / NOT BUY
    confidence = Column(Float, nullable=False)
    model_version = Column(String(50), nullable=False)
    shap_drivers = Column(JSON, nullable=True)
    generated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    def __repr__(self):
        return f"<Prediction {self.ticker}: {self.prediction} ({self.confidence:.2f})>"
