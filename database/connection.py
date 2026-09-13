import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker
from config import DATABASE_URL, SQLITE_FALLBACK_URL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Database")

Base = declarative_base()

def create_active_engine():
    """Tries PostgreSQL first; if connection fails or credentials aren't set, falls back to SQLite."""
    try:
        logger.info(f"Attempting connection to primary PostgreSQL database...")
        pg_engine = create_engine(DATABASE_URL, pool_pre_ping=True)
        with pg_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("Connected to PostgreSQL successfully.")
        return pg_engine
    except Exception as e:
        logger.warning(
            f"PostgreSQL connection failed ({e}). Falling back to local SQLite at {SQLITE_FALLBACK_URL}."
        )
        sqlite_engine = create_engine(
            SQLITE_FALLBACK_URL,
            connect_args={"check_same_thread": False},
        )
        return sqlite_engine

engine = create_active_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Initializes tables if they do not exist."""
    import database.models  # Ensure models are imported before creating tables
    Base.metadata.create_all(bind=engine)
    logger.info("Database schema verified and tables initialized.")
