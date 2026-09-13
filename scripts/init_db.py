import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
from database.connection import init_db, SessionLocal
from database.models import Company
from config import DEFAULT_TICKERS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("InitDB")

INITIAL_COMPANIES = [
    {
        "ticker": "NVDA",
        "company_name": "NVIDIA Corporation",
        "sector": "Technology",
        "industry": "Semiconductors",
    },
    {
        "ticker": "AAPL",
        "company_name": "Apple Inc.",
        "sector": "Technology",
        "industry": "Consumer Electronics",
    },
    {
        "ticker": "MSFT",
        "company_name": "Microsoft Corporation",
        "sector": "Technology",
        "industry": "Software - Infrastructure",
    },
    {
        "ticker": "AMZN",
        "company_name": "Amazon.com, Inc.",
        "sector": "Consumer Cyclical",
        "industry": "Internet Retail",
    },
    {
        "ticker": "GOOGL",
        "company_name": "Alphabet Inc.",
        "sector": "Communication Services",
        "industry": "Internet Content & Information",
    },
]


def seed_companies():
    db = SessionLocal()
    try:
        for comp_data in INITIAL_COMPANIES:
            existing = db.query(Company).filter(Company.ticker == comp_data["ticker"]).first()
            if not existing:
                company = Company(**comp_data)
                db.add(company)
                logger.info(f"Seeded company: {comp_data['ticker']} - {comp_data['company_name']}")
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Error seeding companies: {e}")
    finally:
        db.close()


if __name__ == "__main__":
    logger.info("Initializing database schema...")
    init_db()
    seed_companies()
    logger.info("Database initialization completed successfully.")
