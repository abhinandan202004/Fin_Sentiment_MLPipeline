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


class TrainingFeature(Base):
    __tablename__ = "training_features"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    ticker = Column(String(20), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)

    # Sentiment Features
    avg_sentiment = Column(Float, nullable=False, default=0.0)
    sentiment_ema_3 = Column(Float, nullable=False, default=0.0)
    sentiment_ema_7 = Column(Float, nullable=False, default=0.0)
    sentiment_delta = Column(Float, nullable=False, default=0.0)
    positive_article_count = Column(BigInteger, nullable=False, default=0)
    negative_article_count = Column(BigInteger, nullable=False, default=0)
    total_article_count = Column(BigInteger, nullable=False, default=0)

    # Momentum
    rsi = Column(Float, nullable=True)

    # Trend
    macd = Column(Float, nullable=True)
    macd_signal = Column(Float, nullable=True)
    macd_diff = Column(Float, nullable=True)
    ema20 = Column(Float, nullable=True)
    ema50 = Column(Float, nullable=True)

    # Volatility
    bollinger_high = Column(Float, nullable=True)
    bollinger_low = Column(Float, nullable=True)
    bollinger_pband = Column(Float, nullable=True)
    atr = Column(Float, nullable=True)

    # Price Momentum & Volume
    return_1d = Column(Float, nullable=True)
    return_5d = Column(Float, nullable=True)
    return_20d = Column(Float, nullable=True)
    volume_ratio = Column(Float, nullable=True)

    # Market Regime
    spy_return_1d = Column(Float, nullable=True)
    spy_return_5d = Column(Float, nullable=True)
    spy_volatility = Column(Float, nullable=True)
    vix_level = Column(Float, nullable=True)
    sector_return_5d = Column(Float, nullable=True)

    # Forward Targets
    future_return_1d = Column(Float, nullable=True)
    future_return_5d = Column(Float, nullable=True)
    future_return_10d = Column(Float, nullable=True)
    target = Column(BigInteger, nullable=False)  # 1 if future_return_5d > 0 else 0

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        UniqueConstraint("ticker", "date", name="uq_training_ticker_date"),
        Index("idx_training_ticker_date", "ticker", "date"),
    )

    def __repr__(self):
        return f"<TrainingFeature {self.ticker} {self.date} Target={self.target}>"


class PortfolioHistory(Base):
    __tablename__ = "portfolio_history"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    total_capital = Column(Float, nullable=False)
    cash_weight = Column(Float, nullable=False, default=0.10)
    positions_json = Column(JSON, nullable=False)  # {"NVDA": 0.18, "AAPL": 0.15, ...}

    def __repr__(self):
        return f"<PortfolioHistory {self.timestamp} Capital={self.total_capital}>"


class DecisionHistory(Base):
    __tablename__ = "decision_history"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    ticker = Column(String(20), nullable=False, index=True)
    action = Column(String(20), nullable=False)  # BUY, SELL, HOLD, REDUCE
    target_weight = Column(Float, nullable=False)
    delta_weight = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    expected_return = Column(Float, nullable=True)
    realized_return = Column(Float, nullable=True)
    rationale_json = Column(JSON, nullable=True)  # ["reason 1", "reason 2", ...]

    def __repr__(self):
        return f"<DecisionHistory {self.ticker}: {self.action} ({self.target_weight:.2%})>"

