"""
Backward-compatible proxy module for market ingestion.
Routes directly to ingestion.market.yfinance_ingestor.
"""
from ingestion.market.yfinance_ingestor import YFinanceIngestor, fetch_and_store_market_data

__all__ = ["YFinanceIngestor", "fetch_and_store_market_data"]
