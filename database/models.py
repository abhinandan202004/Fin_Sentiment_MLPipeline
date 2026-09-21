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
    Boolean,
    ForeignKey,
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


class CommitteeDecision(Base):
    __tablename__ = "committee_decisions"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    decision_date = Column(Date, default=lambda: datetime.now(timezone.utc).date())
    prediction_date = Column(Date, default=lambda: datetime.now(timezone.utc).date())
    outcome_date = Column(Date, nullable=True)

    market_regime = Column(String(30), nullable=True)  # Bull, Bear, Sideways, High Volatility, Low Volatility
    regime_confidence = Column(Float, nullable=True)

    ticker = Column(String(20), nullable=False, index=True)
    decision = Column(String(20), nullable=False)  # STRONG BUY, BUY, WATCHLIST, HOLD, REDUCE, SELL, REJECTED
    confidence = Column(Float, nullable=False)
    consensus_score = Column(Float, nullable=False, default=0.0)
    allocation = Column(Float, nullable=False, default=0.0)

    bull_score = Column(Float, nullable=False, default=0.0)
    bear_score = Column(Float, nullable=False, default=0.0)
    risk_score = Column(Float, nullable=False, default=0.0)
    evidence_score = Column(Float, nullable=False, default=0.0)
    final_score = Column(Float, nullable=False, default=0.0)

    evidence_quality = Column(String(20), nullable=False, default="MEDIUM")
    evidence_agreement_score = Column(Float, nullable=True, default=0.50)
    model_probability = Column(Float, nullable=True)
    analog_win_rate = Column(Float, nullable=True)
    governance_passed = Column(Boolean, nullable=False, default=True)
    cio_rationale = Column(Text, nullable=True)

    expected_return = Column(Float, nullable=True)
    realized_return = Column(Float, nullable=True)
    realized_return_5d = Column(Float, nullable=True)
    realized_return_10d = Column(Float, nullable=True)
    realized_return_20d = Column(Float, nullable=True)
    max_drawdown_5d = Column(Float, nullable=True)  # Max intraday drawdown in 5d window
    post_decision_volatility = Column(Float, nullable=True)  # Realized vol 5d after decision
    correct_direction = Column(Boolean, nullable=True)

    def __repr__(self):
        return f"<CommitteeDecision {self.ticker}: {self.decision} (Alloc={self.allocation:.2%}, Conf={self.confidence:.2f})>"


class CommitteeVote(Base):
    __tablename__ = "committee_votes"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    decision_id = Column(String(36), ForeignKey("committee_decisions.id"), nullable=False, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    ticker = Column(String(20), nullable=False, index=True)
    agent_name = Column(String(50), nullable=False)  # BullAnalyst, BearAnalyst, RiskOfficer, EvidenceProsecutor, PortfolioManager
    stance = Column(String(20), nullable=False)
    confidence = Column(Float, nullable=False, default=0.5)
    arguments_json = Column(JSON, nullable=True)
    is_accurate = Column(Boolean, nullable=True)

    def __repr__(self):
        return f"<CommitteeVote {self.agent_name} for {self.ticker}: {self.stance} ({self.confidence:.2f})>"


class OpportunityHistory(Base):
    __tablename__ = "opportunity_history"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    discovery_date = Column(Date, default=lambda: datetime.now(timezone.utc).date(), index=True)
    ticker = Column(String(20), nullable=False, index=True)
    score = Column(Float, nullable=False)  # 0 - 100
    confidence = Column(Float, nullable=False)  # 0.0 - 1.0
    ml_probability = Column(Float, nullable=True)
    analog_win_rate = Column(Float, nullable=True)
    sentiment_rank = Column(Float, nullable=True)
    sector_momentum = Column(Float, nullable=True)
    event_impact = Column(Float, nullable=True)
    market_regime = Column(String(30), nullable=True)
    catalyst_summary = Column(Text, nullable=True)
    trend_theme = Column(String(50), nullable=True)
    priority = Column(BigInteger, default=1)
    status = Column(String(20), default="ACTIVE")  # ACTIVE, RESOLVED, EXPIRED

    # Multi-horizon outcome tracking fields
    return_5d = Column(Float, nullable=True)
    return_20d = Column(Float, nullable=True)
    return_60d = Column(Float, nullable=True)
    max_drawdown = Column(Float, nullable=True)
    success = Column(Boolean, nullable=True)
    resolved_date = Column(Date, nullable=True)

    def __repr__(self):
        return f"<OpportunityHistory {self.ticker}: Score={self.score:.1f} (Pri={self.priority})>"


class TrendHistory(Base):
    __tablename__ = "trend_history"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    detected_date = Column(Date, default=lambda: datetime.now(timezone.utc).date(), index=True)
    theme = Column(String(50), nullable=False, index=True)  # AI, Cybersecurity, Cloud, etc.
    mention_growth = Column(Float, nullable=False, default=0.0)  # % growth in mentions
    sentiment_trend = Column(Float, nullable=False, default=0.0)  # Sentiment polarity change
    news_volume = Column(BigInteger, nullable=False, default=0)
    sector_strength = Column(Float, nullable=False, default=0.0)
    leading_tickers = Column(JSON, nullable=True)  # ["NVDA", "MSFT", ...]

    def __repr__(self):
        return f"<TrendHistory {self.theme}: Growth={self.mention_growth:+.1%}>"


class CatalystHistory(Base):
    __tablename__ = "catalyst_history"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    detected_date = Column(Date, default=lambda: datetime.now(timezone.utc).date(), index=True)
    ticker = Column(String(20), nullable=False, index=True)
    catalyst_type = Column(String(50), nullable=False)  # Earnings, Product Launches, etc.
    description = Column(Text, nullable=False)
    expected_impact = Column(Float, nullable=False, default=0.0)  # -1.0 to +1.0
    event_date = Column(Date, nullable=True)
    source_url = Column(Text, nullable=True)

    def __repr__(self):
        return f"<CatalystHistory {self.ticker}: {self.catalyst_type} (Impact={self.expected_impact:+.2f})>"


