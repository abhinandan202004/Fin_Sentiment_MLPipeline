import logging
from datetime import datetime, date
import numpy as np
import pandas as pd
import yfinance as yf
from sqlalchemy.orm import Session
from database.connection import SessionLocal
from database.models import MarketData, Company

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MarketIngestion")


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Computes RSI (14), MACD (12, 26, 9), Volatility (20d), and Forward Returns."""
    df = df.copy()
    close = df["Close"]

    # 1. RSI (14)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))

    # 2. MACD (12, 26, 9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # 3. Volatility (Annualized 20-day rolling std of log returns)
    log_ret = np.log(close / close.shift(1))
    df["volatility"] = log_ret.rolling(window=20).std() * np.sqrt(252)

    # 4. Forward Returns (for Supervised Prediction Target)
    df["return_1d"] = (close.shift(-1) - close) / close
    df["return_5d"] = (close.shift(-5) - close) / close
    df["return_10d"] = (close.shift(-10) - close) / close

    return df


def fetch_and_store_market_data(
    ticker: str,
    period: str = "2y",
    db: Session = None,
) -> int:
    """
    Pulls OHLCV bars from Yahoo Finance, computes technical indicators,
    and updates/stores records in the market_data database table.
    """
    should_close_db = False
    if db is None:
        db = SessionLocal()
        should_close_db = True

    try:
        logger.info(f"Fetching market data for {ticker} over period={period}...")
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)

        if hist.empty:
            logger.warning(f"No price data found for ticker {ticker}.")
            return 0

        # Drop any empty or all-NaN price rows from yfinance
        hist = hist.dropna(subset=["Open", "High", "Low", "Close"]).copy()
        if hist.empty:
            logger.warning(f"No valid price bars after dropping NaNs for {ticker}.")
            return 0

        # Also store or update Company profile if available
        try:
            info = stock.info or {}
            existing_company = db.query(Company).filter(Company.ticker == ticker).first()
            if not existing_company:
                company = Company(
                    ticker=ticker,
                    company_name=info.get("shortName") or info.get("longName") or ticker,
                    sector=info.get("sector"),
                    industry=info.get("industry"),
                    market_cap=info.get("marketCap"),
                )
                db.add(company)
                db.commit()
        except Exception as e:
            logger.debug(f"Company profile metadata fetch skipped for {ticker}: {e}")
            db.rollback()

        # Compute Technical Indicators
        df = compute_technical_indicators(hist)

        # Batch upsert into database
        records_saved = 0
        for dt_index, row in df.iterrows():
            trade_date = dt_index.date() if isinstance(dt_index, pd.Timestamp) else dt_index

            existing = (
                db.query(MarketData)
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
                existing.macd_signal = (
                    None if pd.isna(row["macd_signal"]) else float(row["macd_signal"])
                )
                existing.volatility = (
                    None if pd.isna(row["volatility"]) else float(row["volatility"])
                )
                existing.return_1d = (
                    None if pd.isna(row["return_1d"]) else float(row["return_1d"])
                )
                existing.return_5d = (
                    None if pd.isna(row["return_5d"]) else float(row["return_5d"])
                )
                existing.return_10d = (
                    None if pd.isna(row["return_10d"]) else float(row["return_10d"])
                )
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
                db.add(entry)
            records_saved += 1

        db.commit()
        logger.info(f"Successfully processed {records_saved} market data bars for {ticker}.")
        return records_saved

    except Exception as e:
        db.rollback()
        logger.error(f"Error fetching market data for {ticker}: {e}")
        raise
    finally:
        if should_close_db:
            db.close()
