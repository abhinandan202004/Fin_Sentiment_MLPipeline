import logging
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
import yfinance as yf
from sqlalchemy.orm import Session
from database.connection import SessionLocal
from database.models import MarketData, Company

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("YFinanceIngestor")


class YFinanceIngestor:
    """
    Ingests market price bars (OHLCV) and company metadata from Yahoo Finance.
    Supports configurable periods ('1mo', '6mo', '1y', '2y', '5y', 'max')
    and intervals ('1d', '1wk', '1h').
    """

    def __init__(self, db: Optional[Session] = None):
        self._db = db

    def get_session(self) -> Session:
        return self._db if self._db is not None else SessionLocal()

    def fetch_ohlcv(
        self,
        ticker: str,
        period: str = "2y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetches and cleans OHLCV history for a ticker."""
        logger.info(f"Fetching OHLCV for {ticker} (period={period}, interval={interval})...")
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)

        if df.empty:
            logger.warning(f"No price data found for {ticker}.")
            return pd.DataFrame()

        # Ensure essential columns exist and clean NaNs
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        for col in required_cols:
            if col not in df.columns:
                logger.error(f"Missing required price column {col} in yfinance response.")
                return pd.DataFrame()

        df = df.dropna(subset=required_cols).copy()
        return df

    def upsert_company_metadata(self, ticker: str, db: Session) -> None:
        """Fetches and updates company metadata (name, sector, industry, market_cap)."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info or {}
            existing = db.query(Company).filter(Company.ticker == ticker).first()
            name = info.get("shortName") or info.get("longName") or ticker
            sector = info.get("sector")
            industry = info.get("industry")
            market_cap = info.get("marketCap")

            if existing:
                if sector:
                    existing.sector = sector
                if industry:
                    existing.industry = industry
                if market_cap:
                    existing.market_cap = market_cap
            else:
                comp = Company(
                    ticker=ticker,
                    company_name=name,
                    sector=sector,
                    industry=industry,
                    market_cap=market_cap,
                )
                db.add(comp)
            db.commit()
        except Exception as e:
            db.rollback()
            logger.debug(f"Company metadata update skipped for {ticker}: {e}")

    def ingest_ticker(
        self,
        ticker: str,
        period: str = "2y",
        interval: str = "1d",
        db: Optional[Session] = None,
    ) -> int:
        """Fetches OHLCV, updates metadata, and upserts bars into market_data table."""
        session = db if db is not None else self.get_session()
        should_close = db is None and self._db is None

        try:
            # 1. Upsert company profile
            self.upsert_company_metadata(ticker, session)

            # 2. Fetch raw OHLCV
            df = self.fetch_ohlcv(ticker, period=period, interval=interval)
            if df.empty:
                return 0

            # 3. Compute baseline technical indicators for market_data table
            close = df["Close"]
            delta = close.diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
            avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
            rs = avg_gain / (avg_loss + 1e-9)
            df["rsi"] = 100 - (100 / (1 + rs))

            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            df["macd"] = ema12 - ema26
            df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

            log_ret = np.log(close / close.shift(1))
            df["volatility"] = log_ret.rolling(window=20).std() * np.sqrt(252)

            df["return_1d"] = (close.shift(-1) - close) / close
            df["return_5d"] = (close.shift(-5) - close) / close
            df["return_10d"] = (close.shift(-10) - close) / close

            # 4. Upsert into database
            count = 0
            for dt_index, row in df.iterrows():
                trade_date = dt_index.date() if isinstance(dt_index, pd.Timestamp) else dt_index

                existing = (
                    session.query(MarketData)
                    .filter(MarketData.ticker == ticker, MarketData.date == trade_date)
                    .first()
                )

                if existing:
                    existing.open = float(row["Open"])
                    existing.high = float(row["High"])
                    existing.low = float(row["Low"])
                    existing.close = float(row["Close"])
                    existing.volume = int(row["Volume"])
                    existing.rsi = None if pd.isna(row["rsi"]) else float(row["rsi"])
                    existing.macd = None if pd.isna(row["macd"]) else float(row["macd"])
                    existing.macd_signal = None if pd.isna(row["macd_signal"]) else float(row["macd_signal"])
                    existing.volatility = None if pd.isna(row["volatility"]) else float(row["volatility"])
                    existing.return_1d = None if pd.isna(row["return_1d"]) else float(row["return_1d"])
                    existing.return_5d = None if pd.isna(row["return_5d"]) else float(row["return_5d"])
                    existing.return_10d = None if pd.isna(row["return_10d"]) else float(row["return_10d"])
                else:
                    entry = MarketData(
                        ticker=ticker,
                        date=trade_date,
                        open=float(row["Open"]),
                        high=float(row["High"]),
                        low=float(row["Low"]),
                        close=float(row["Close"]),
                        volume=int(row["Volume"]),
                        rsi=None if pd.isna(row["rsi"]) else float(row["rsi"]),
                        macd=None if pd.isna(row["macd"]) else float(row["macd"]),
                        macd_signal=None if pd.isna(row["macd_signal"]) else float(row["macd_signal"]),
                        volatility=None if pd.isna(row["volatility"]) else float(row["volatility"]),
                        return_1d=None if pd.isna(row["return_1d"]) else float(row["return_1d"]),
                        return_5d=None if pd.isna(row["return_5d"]) else float(row["return_5d"]),
                        return_10d=None if pd.isna(row["return_10d"]) else float(row["return_10d"]),
                    )
                    session.add(entry)
                count += 1

            session.commit()
            logger.info(f"Ingested {count} bars for {ticker}.")
            return count

        except Exception as e:
            session.rollback()
            logger.error(f"Error ingesting {ticker}: {e}")
            raise
        finally:
            if should_close:
                session.close()


def fetch_and_store_market_data(
    ticker: str,
    period: str = "2y",
    interval: str = "1d",
    db: Optional[Session] = None,
) -> int:
    """Convenience functional wrapper maintaining full backward compatibility."""
    ingestor = YFinanceIngestor(db=db)
    return ingestor.ingest_ticker(ticker, period=period, interval=interval, db=db)
