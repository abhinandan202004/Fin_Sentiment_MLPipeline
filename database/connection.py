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
        pg_engine = create_engine(DATABASE_URL, pool_pre_ping=True, connect_args={"connect_timeout": 2})
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
    """Initializes tables if they do not exist, ensuring updated schema."""
    from sqlalchemy import inspect
    import database.models  # Ensure models are imported before creating tables
    inspector = inspect(engine)
    if "training_features" in inspector.get_table_names():
        existing_cols = {col["name"] for col in inspector.get_columns("training_features")}
        if "spy_return_1d" not in existing_cols or "vix_level" not in existing_cols:
            logger.info("Migrating training_features table schema (dropping legacy table)...")
            database.models.TrainingFeature.__table__.drop(bind=engine, checkfirst=True)
    if "committee_decisions" in inspector.get_table_names():
        existing_cols = {col["name"] for col in inspector.get_columns("committee_decisions")}
        for col_name in ["model_probability", "analog_win_rate", "evidence_agreement_score", "realized_return_5d", "realized_return_20d"]:
            if col_name not in existing_cols:
                logger.info(f"Adding missing column {col_name} to committee_decisions...")
                try:
                    with engine.connect() as conn:
                        conn.execute(text(f"ALTER TABLE committee_decisions ADD COLUMN {col_name} FLOAT;"))
                        conn.commit()
                except Exception as e:
                    logger.warning(f"Could not add column {col_name}: {e}")
    Base.metadata.create_all(bind=engine)
    logger.info("Database schema verified and tables initialized.")
