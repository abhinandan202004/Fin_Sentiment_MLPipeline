"""
Opportunity Scanner for Sprint 8: Autonomous Research & Market Opportunity Discovery.

Scans the coverage universe daily across all 25 tracked tickers, evaluating:
  - Calibrated ML conviction probability
  - 30,025 FAISS historical analog win rate & sample power
  - Cross-sectional sentiment percentile
  - Sector momentum, RSI, and technical squeeze state
  - Corporate NLP event impact
  - Market macro regime

Persists all discoveries to the `opportunity_history` database table and outputs
a ranked opportunity ranking.

Usage:
    python opportunities/scanner.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone

from database.connection import SessionLocal, init_db
from database.models import OpportunityHistory, TrainingFeature, MarketData
from opportunities.scoring import OpportunityScorer
from agents.analog_agent import AnalogAgent
from agents.technical_agent import TechnicalAgent
from agents.event_agent import EventAgent
from agents.regime_agent import RegimeAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("OpportunityScanner")

DEFAULT_UNIVERSE = [
    "AAPL", "AMD", "AMZN", "BAC", "CAT", "CRM", "CVX", "GOOGL", "GS", "HD",
    "HON", "INTC", "JNJ", "JPM", "META", "MSFT", "NFLX", "NVDA", "ORCL", "PFE",
    "TSLA", "UNH", "V", "WMT", "XOM"
]


class OpportunityScanner:
    """Automated daily multi-factor scanner across the equity coverage universe."""

    def __init__(self, universe: Optional[List[str]] = None):
        self.universe = universe or DEFAULT_UNIVERSE
        self.scorer = OpportunityScorer()
        self.analog_agent = AnalogAgent()
        self.tech_agent = TechnicalAgent()
        self.event_agent = EventAgent()
        self.regime_agent = RegimeAgent()

        # Lazy load PredictionService to handle environments without full models cleanly
        self._prediction_service = None

    @property
    def prediction_service(self):
        if self._prediction_service is None:
            try:
                from models.predict import PredictionService
                self._prediction_service = PredictionService()
            except Exception as e:
                logger.warning(f"Could not load full PredictionService ({e}). Using feature store fallback.")
                self._prediction_service = False
        return self._prediction_service

    def get_ticker_features(self, ticker: str) -> Dict[str, Any]:
        """Fetches the latest quantitative features from the DB feature store."""
        session = SessionLocal()
        try:
            row = (
                session.query(TrainingFeature)
                .filter(TrainingFeature.ticker == ticker)
                .order_by(TrainingFeature.date.desc())
                .first()
            )
            if row:
                return {
                    "rsi": row.rsi if row.rsi is not None else 50.0,
                    "avg_sentiment": row.avg_sentiment if row.avg_sentiment is not None else 0.0,
                    "sector_return_5d": row.sector_return_5d if row.sector_return_5d is not None else 0.0,
                    "vix_level": row.vix_level if row.vix_level is not None else 18.0,
                    "volume_ratio": row.volume_ratio if row.volume_ratio is not None else 1.0
                }
            return {
                "rsi": 50.0,
                "avg_sentiment": 0.0,
                "sector_return_5d": 0.0,
                "vix_level": 18.0,
                "volume_ratio": 1.0
            }
        finally:
            session.close()

    def scan_ticker(self, ticker: str, market_regime: str = "Neutral") -> Dict[str, Any]:
        """Evaluates all evidence streams for a single ticker and returns the opportunity score."""
        ticker = ticker.upper().strip()
        features = self.get_ticker_features(ticker)

        # 1. ML Probability
        ml_prob = 0.50
        ml_threshold = 0.55
        if self.prediction_service:
            try:
                pred = self.prediction_service.predict(ticker)
                ml_prob = float(pred.get("probability", 0.50))
                ml_threshold = float(pred.get("threshold", 0.55))
            except Exception as e:
                logger.debug(f"Prediction failed for {ticker}: {e}")
                ml_prob = 0.52 if features["avg_sentiment"] > 0 else 0.48

        # 2. Historical Analogs (30k FAISS)
        analog_win = 0.50
        analog_samples = 50
        ci_lower = 0.0
        try:
            analogs = self.analog_agent.analyze_analogs(ticker, top_k=50)
            analog_win = float(analogs.get("success_rate", 0.50))
            analog_samples = int(analogs.get("sample_size", 50))
            ci = analogs.get("confidence_interval", [-0.01, 0.02])
            ci_lower = float(ci[0]) if isinstance(ci, (list, tuple)) and len(ci) > 0 else 0.0
        except Exception as e:
            logger.debug(f"Analog lookup failed for {ticker}: {e}")

        # 3. Sentiment Percentile Rank
        sentiment_rank = 0.50
        if features["avg_sentiment"] > 0.3:
            sentiment_rank = 0.80
        elif features["avg_sentiment"] > 0.0:
            sentiment_rank = 0.60
        elif features["avg_sentiment"] < -0.3:
            sentiment_rank = 0.20
        else:
            sentiment_rank = 0.40

        # 4. Technical / Sector Momentum
        rsi = features["rsi"]
        sec_ret = features["sector_return_5d"]
        squeeze = "VOLATILITY_EXPANDING"
        try:
            tech = self.tech_agent.analyze_ticker(ticker)
            squeeze = tech.get("squeeze_state", "VOLATILITY_EXPANDING")
            if tech.get("rsi") is not None:
                rsi = float(tech["rsi"])
        except Exception as e:
            logger.debug(f"Tech analysis failed for {ticker}: {e}")

        # 5. Events Impact
        event_impact = 0.0
        has_catalyst = False
        try:
            raw_events = self.event_agent.detect_events_for_ticker(ticker, limit=3)
            if raw_events:
                has_catalyst = True
                impacts = [e.get("impact", 0.0) for e in raw_events]
                event_impact = float(sum(impacts) / len(impacts))
        except Exception as e:
            logger.debug(f"Event detection failed for {ticker}: {e}")

        # Compute multi-factor opportunity score
        res = self.scorer.compute_opportunity(
            ticker=ticker,
            ml_prob=ml_prob,
            ml_threshold=ml_threshold,
            analog_win_rate=analog_win,
            analog_sample_size=analog_samples,
            analog_ci_lower=ci_lower,
            sentiment_rank=sentiment_rank,
            raw_sentiment=features["avg_sentiment"],
            sector_return_5d=sec_ret,
            rsi=rsi,
            squeeze_state=squeeze,
            event_impact=event_impact,
            has_recent_catalyst=has_catalyst,
            market_regime=market_regime
        )

        res.update({
            "ml_probability": round(ml_prob, 4),
            "analog_win_rate": round(analog_win, 4),
            "sentiment_rank": round(sentiment_rank, 4),
            "sector_momentum": round(sec_ret, 4),
            "event_impact": round(event_impact, 4),
            "rsi": round(rsi, 1)
        })
        return res

    def scan_universe(self, save_to_db: bool = True) -> Dict[str, Any]:
        """
        Executes full multi-factor scan across the coverage universe.
        Ranks opportunities descending by score and persists to opportunity_history.
        """
        init_db()
        logger.info(f"Scanning {len(self.universe)} tickers across coverage universe...")

        # Detect current market regime
        regime = "Neutral"
        try:
            reg_res = self.regime_agent.detect_regime()
            regime = reg_res.get("regime", "Neutral")
        except Exception as e:
            logger.warning(f"Regime detection failed: {e}")

        results = []
        for ticker in self.universe:
            opp = self.scan_ticker(ticker, market_regime=regime)
            results.append(opp)

        # Rank descending by opportunity_score
        results.sort(key=lambda x: x["opportunity_score"], reverse=True)

        # Assign priorities
        for i, opp in enumerate(results):
            opp["priority"] = i + 1

        top_opportunities = [r["ticker"] for r in results if r["opportunity_score"] >= 65.0]
        if not top_opportunities:
            top_opportunities = [r["ticker"] for r in results[:3]]

        # Persist to database
        if save_to_db:
            self._persist_opportunities(results, regime)

        output = {
            "discovery_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "market_regime": regime,
            "total_scanned": len(results),
            "top_opportunities": top_opportunities[:5],
            "opportunities": results
        }
        return output

    def _persist_opportunities(self, opportunities: List[Dict[str, Any]], regime: str):
        """Saves scanned opportunities to the opportunity_history table."""
        session = SessionLocal()
        today = datetime.now(timezone.utc).date()
        try:
            for opp in opportunities:
                rec = OpportunityHistory(
                    discovery_date=today,
                    ticker=opp["ticker"],
                    score=opp["opportunity_score"],
                    confidence=opp["confidence"],
                    ml_probability=opp.get("ml_probability"),
                    analog_win_rate=opp.get("analog_win_rate"),
                    sentiment_rank=opp.get("sentiment_rank"),
                    sector_momentum=opp.get("sector_momentum"),
                    event_impact=opp.get("event_impact"),
                    market_regime=regime,
                    priority=opp.get("priority", 1),
                    status="ACTIVE"
                )
                session.add(rec)
            session.commit()
            logger.info(f"Persisted {len(opportunities)} discovered opportunities to opportunity_history.")
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to persist opportunities: {e}")
        finally:
            session.close()


if __name__ == "__main__":
    scanner = OpportunityScanner()
    report = scanner.scan_universe(save_to_db=True)
    
    print("\n" + "=" * 65)
    print("Sprint 8: Autonomous Market Opportunity Scanner")
    print(f"Date: {report['discovery_date']} | Regime: {report['market_regime']} | Scanned: {report['total_scanned']}")
    print("=" * 65)
    print(f"{'Rank':<5} {'Ticker':<8} {'Score':<8} {'Confidence':<12} {'ML Prob':<10} {'Analog Win':<12}")
    print("-" * 65)
    for opp in report["opportunities"][:10]:
        print(f"{opp['priority']:<5} {opp['ticker']:<8} {opp['opportunity_score']:<8.1f} {opp['confidence']:<12.1%} {opp['ml_probability']:<10.1%} {opp['analog_win_rate']:<12.1%}")
    print("=" * 65)
    print(f"Top Recommended Opportunities: {', '.join(report['top_opportunities'])}")
