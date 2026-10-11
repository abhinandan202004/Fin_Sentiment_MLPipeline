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


# =====================================================================
# SPRINT 9 — AUTONOMOUS EVALUATION & ADAPTIVE INTELLIGENCE MODELS
# =====================================================================

class PredictionOutcome(Base):
    __tablename__ = "prediction_outcomes"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    prediction_id = Column(String(50), nullable=True, index=True)
    ticker = Column(String(20), nullable=False, index=True)
    prediction_date = Column(DateTime, nullable=False, index=True)
    predicted_probability = Column(Float, nullable=False)
    threshold = Column(Float, nullable=False)
    predicted_direction = Column(String(10), nullable=False, default="BUY")
    status = Column(String(20), nullable=False, default="PENDING", index=True)  # PENDING, RESOLVED, UNRESOLVABLE
    entry_price = Column(Float, nullable=True)
    exit_price_5d = Column(Float, nullable=True)
    max_drawdown = Column(Float, nullable=True)
    horizon_status = Column(JSON, nullable=True)  # {"1d": "RESOLVED", "5d": "PENDING", ...}

    actual_return_1d = Column(Float, nullable=True)
    actual_return_5d = Column(Float, nullable=True)
    actual_return_10d = Column(Float, nullable=True)
    actual_return_20d = Column(Float, nullable=True)
    actual_return_60d = Column(Float, nullable=True)

    actual_direction = Column(String(10), nullable=True)

    correct_1d = Column(Boolean, nullable=True)
    correct_5d = Column(Boolean, nullable=True)
    correct_10d = Column(Boolean, nullable=True)
    correct_20d = Column(Boolean, nullable=True)
    correct_60d = Column(Boolean, nullable=True)

    brier_score = Column(Float, nullable=True)

    market_regime = Column(String(50), nullable=True)
    sector = Column(String(50), nullable=True)

    model_version = Column(String(50), nullable=True, default="SPRINT_9_BASELINE")
    feature_version = Column(String(50), nullable=True, default="v2.0_43_features")
    threshold_version = Column(String(50), nullable=True, default="calibrated_0.60")
    calibration_version = Column(String(50), nullable=True, default="raw_uncalibrated")

    resolved_date = Column(DateTime, nullable=True)

    __table_args__ = (
        Index("idx_pred_outcome_ticker_date", "ticker", "prediction_date"),
    )

    def __repr__(self):
        return f"<PredictionOutcome {self.ticker} {self.prediction_date.strftime('%Y-%m-%d') if self.prediction_date else ''}: prob={self.predicted_probability:.3f}>"


class AgentDecisionRecord(Base):
    __tablename__ = "agent_decision_records"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    agent_name = Column(String(50), nullable=False, index=True)
    ticker = Column(String(20), nullable=False, index=True)
    stance = Column(String(20), nullable=False)
    confidence = Column(Float, nullable=False)
    predicted_probability = Column(Float, nullable=True)
    predicted_direction = Column(String(10), nullable=True)

    market_regime = Column(String(50), nullable=True)
    sector = Column(String(50), nullable=True)

    committee_decision_id = Column(String(50), nullable=True, index=True)

    model_version = Column(String(50), nullable=True, default="SPRINT_9_BASELINE")
    committee_version = Column(String(50), nullable=True, default="v1.0_baseline")

    def __repr__(self):
        return f"<AgentDecisionRecord {self.agent_name} {self.ticker}: {self.stance} ({self.confidence:.2f})>"


class AgentEvaluationRecord(Base):
    __tablename__ = "agent_evaluation_records"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    evaluation_date = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    agent_name = Column(String(50), nullable=False, index=True)
    sample_size = Column(BigInteger, nullable=False, default=0)

    directional_accuracy = Column(Float, nullable=True)
    hit_rate = Column(Float, nullable=True)
    brier_score = Column(Float, nullable=True)
    ece = Column(Float, nullable=True)

    average_return = Column(Float, nullable=True)
    median_return = Column(Float, nullable=True)
    sharpe_ratio = Column(Float, nullable=True)
    sortino_ratio = Column(Float, nullable=True)
    max_drawdown = Column(Float, nullable=True)

    avg_drawdown_on_miss = Column(Float, nullable=True)

    regime_breakdown = Column(JSON, nullable=True)
    sector_breakdown = Column(JSON, nullable=True)
    confidence_breakdown = Column(JSON, nullable=True)

    def __repr__(self):
        return f"<AgentEvaluationRecord {self.agent_name}: n={self.sample_size} hit_rate={self.hit_rate:.1%}>"


class CommitteeEvaluationRecord(Base):
    __tablename__ = "committee_evaluation_records"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    evaluation_date = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    decisions_evaluated = Column(BigInteger, nullable=False, default=0)
    accuracy = Column(Float, nullable=True)
    hit_rate = Column(Float, nullable=True)

    avg_return_5d = Column(Float, nullable=True)
    avg_return_20d = Column(Float, nullable=True)
    avg_return_60d = Column(Float, nullable=True)

    max_drawdown = Column(Float, nullable=True)
    sharpe_ratio = Column(Float, nullable=True)
    sortino_ratio = Column(Float, nullable=True)
    profit_factor = Column(Float, nullable=True)

    edge_vs_ml = Column(Float, nullable=True)
    edge_vs_consensus = Column(Float, nullable=True)
    edge_vs_benchmark = Column(Float, nullable=True)

    vetoes_evaluated = Column(BigInteger, nullable=False, default=0)
    vetoes_prevented_losses = Column(BigInteger, nullable=False, default=0)

    committee_version = Column(String(50), nullable=True, default="SPRINT_9_BASELINE")

    def __repr__(self):
        return f"<CommitteeEvaluationRecord eval_date={self.evaluation_date} n={self.decisions_evaluated}>"


class OpportunityOutcome(Base):
    __tablename__ = "opportunity_outcomes"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    opportunity_id = Column(String(50), nullable=True, index=True)
    ticker = Column(String(20), nullable=False, index=True)
    discovery_date = Column(DateTime, nullable=False, index=True)
    opportunity_score = Column(Float, nullable=False)
    status = Column(String(20), nullable=False, default="PENDING", index=True)  # PENDING, RESOLVED, UNRESOLVABLE
    entry_price = Column(Float, nullable=True)
    max_drawdown = Column(Float, nullable=True)
    horizon_status = Column(JSON, nullable=True)  # {"1d": "RESOLVED", "5d": "PENDING", ...}

    return_1d = Column(Float, nullable=True)
    return_5d = Column(Float, nullable=True)
    return_10d = Column(Float, nullable=True)
    return_20d = Column(Float, nullable=True)
    return_60d = Column(Float, nullable=True)

    success = Column(Boolean, nullable=True)

    market_regime = Column(String(50), nullable=True)
    sector = Column(String(50), nullable=True)
    resolved_date = Column(DateTime, nullable=True)

    def __repr__(self):
        return f"<OpportunityOutcome {self.ticker}: score={self.opportunity_score:.1f} success={self.success}>"


class PaperTradingPosition(Base):
    __tablename__ = "paper_trading_positions"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    entry_date = Column(DateTime, nullable=False, index=True)
    ticker = Column(String(20), nullable=False, index=True)
    direction = Column(String(10), nullable=False, default="BUY")
    entry_price = Column(Float, nullable=False)
    shares = Column(Float, nullable=False)

    current_price = Column(Float, nullable=False)

    gross_pnl = Column(Float, default=0.0)
    commission = Column(Float, default=0.0)
    slippage_cost = Column(Float, default=0.0)
    spread_cost = Column(Float, default=0.0)
    total_transaction_cost = Column(Float, default=0.0)
    net_pnl = Column(Float, default=0.0)

    unrealized_pnl = Column(Float, default=0.0)
    realized_pnl = Column(Float, default=0.0)

    exit_date = Column(DateTime, nullable=True)
    status = Column(String(20), nullable=False, default="OPEN")  # OPEN, CLOSED

    def __repr__(self):
        return f"<PaperTradingPosition {self.ticker} shares={self.shares:.1f} net_pnl={self.net_pnl:.2f}>"


class PaperTradingSnapshot(Base):
    __tablename__ = "paper_trading_snapshots"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    snapshot_date = Column(DateTime, nullable=False, index=True)

    total_equity = Column(Float, nullable=False)
    cash = Column(Float, nullable=False)
    positions_value = Column(Float, nullable=False)

    daily_return = Column(Float, default=0.0)
    cumulative_return = Column(Float, default=0.0)

    benchmark_return = Column(Float, default=0.0)
    alpha = Column(Float, default=0.0)

    max_drawdown = Column(Float, default=0.0)
    sharpe_ratio = Column(Float, nullable=True)
    sortino_ratio = Column(Float, nullable=True)

    def __repr__(self):
        return f"<PaperTradingSnapshot date={self.snapshot_date} equity={self.total_equity:.2f} alpha={self.alpha:.2%}>"


class ModelCalibrationRecord(Base):
    __tablename__ = "model_calibration_records"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    calibration_date = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    method = Column(String(50), nullable=False)  # Platt, Isotonic
    ece_before = Column(Float, nullable=False)
    ece_after = Column(Float, nullable=False)
    brier_before = Column(Float, nullable=False)
    brier_after = Column(Float, nullable=False)

    optimal_threshold = Column(Float, nullable=True)

    parameters = Column(JSON, nullable=True)

    model_version = Column(String(50), nullable=True, default="SPRINT_9_BASELINE")
    calibration_version = Column(String(50), nullable=True, default="calibrated_v1.0")

    def __repr__(self):
        return f"<ModelCalibrationRecord {self.method} ECE: {self.ece_before:.4f} -> {self.ece_after:.4f}>"


class EvaluationSnapshot(Base):
    __tablename__ = "evaluation_snapshots"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    evaluation_window = Column(String(100), nullable=False)

    model_version = Column(String(50), nullable=True, default="SPRINT_9_BASELINE")
    feature_version = Column(String(50), nullable=True, default="v2.0_43_features")
    threshold_version = Column(String(50), nullable=True, default="calibrated_0.60")
    calibration_version = Column(String(50), nullable=True, default="calibrated_v1.0")
    committee_version = Column(String(50), nullable=True, default="adaptive_v1.0")
    opportunity_version = Column(String(50), nullable=True, default="adaptive_v1.0")

    model_metrics_json = Column(JSON, nullable=True)
    agent_metrics_json = Column(JSON, nullable=True)
    committee_metrics_json = Column(JSON, nullable=True)
    calibration_metrics_json = Column(JSON, nullable=True)
    adaptive_weights_json = Column(JSON, nullable=True)
    opportunity_weights_json = Column(JSON, nullable=True)
    paper_trading_metrics_json = Column(JSON, nullable=True)

    def __repr__(self):
        return f"<EvaluationSnapshot window={self.evaluation_window} time={self.timestamp}>"



