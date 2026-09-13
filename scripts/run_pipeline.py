import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import logging
from datetime import datetime, timezone
import pandas as pd

from config import DEFAULT_TICKERS, MARKET_HISTORY_PERIOD
from database.connection import init_db, SessionLocal
from database.models import Prediction
from ingestion.market import fetch_and_store_market_data
from ingestion.news import fetch_and_store_news
from nlp.sentiment import score_unscored_articles
from features.builder import build_feature_dataset, FEATURE_COLUMNS, TARGET_COLUMN
from models.xgboost_model import train_xgboost_pipeline
from explainability.shap_explainer import (
    get_global_feature_importance,
    explain_instance,
)
from backtest.engine import run_backtest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("MLPipeline")


def run_pipeline(
    tickers: list = None,
    period: str = MARKET_HISTORY_PERIOD,
    skip_news: bool = False,
    confidence_thresh: float = 0.50,
):
    if not tickers:
        tickers = DEFAULT_TICKERS

    print("\n" + "=" * 80)
    print(" FINANCIAL INTELLIGENCE ML RESEARCH PIPELINE (MVP BASELINE)")
    print(f" Target Tickers : {', '.join(tickers)}")
    print(f" Market History : {period}")
    print("=" * 80 + "\n")

    # Step 1: Database Initialization
    logger.info("Step 1/7: Checking and initializing database schema...")
    init_db()

    # Step 2: Market Data Ingestion
    logger.info("Step 2/7: Ingesting market price bars & calculating technical indicators...")
    for t in tickers:
        fetch_and_store_market_data(t, period=period)

    # Step 3: News Ingestion
    if not skip_news:
        logger.info("Step 3/7: Ingesting financial news and headlines...")
        for t in tickers:
            fetch_and_store_news(t)
    else:
        logger.info("Step 3/7: Skipping news ingestion (--skip-news flag set).")

    # Step 4: FinBERT Sentiment Scoring
    logger.info("Step 4/7: Running FinBERT NLP sentiment inference on news articles...")
    scored_count = score_unscored_articles()
    logger.info(f"FinBERT processed {scored_count} articles.")

    # Step 5: Feature Engineering
    logger.info("Step 5/7: Building feature matrix aligning sentiment with market technicals...")
    df_features = build_feature_dataset(tickers=tickers)
    if df_features.empty or len(df_features) < 30:
        logger.error("Insufficient feature samples generated to train model. Pipeline aborted.")
        return

    print(f"\n[Feature Dataset Summary]")
    print(f"  Total Valid Rows  : {len(df_features)}")
    print(f"  Target Distribution: BUY={int(df_features[TARGET_COLUMN].sum())}, NOT BUY={int((df_features[TARGET_COLUMN] == 0).sum())}")
    print(f"  Features Included  : {', '.join(FEATURE_COLUMNS)}\n")

    # Step 6: XGBoost Baseline Model Training
    logger.info("Step 6/7: Training XGBoost binary classification model (Temporal Split)...")
    model, metrics, test_df = train_xgboost_pipeline(df_features)

    print("\n" + "-" * 50)
    print(" MODEL PERFORMANCE METRICS (Out-of-Sample Test)")
    print("-" * 50)
    print(f"  Accuracy       : {metrics['accuracy']:.2%}")
    print(f"  BUY Win Rate   : {metrics['win_rate']:.2%}")
    print(f"  Precision      : {metrics['precision']:.2%}")
    print(f"  Recall         : {metrics['recall']:.2%}")
    print(f"  F1 Score       : {metrics['f1']:.4f}")
    print(f"  ROC-AUC        : {metrics['roc_auc']:.4f}")
    print(f"  Confusion Mat  : {metrics['confusion_matrix']}")
    print("-" * 50 + "\n")

    # Step 7: SHAP Explainability & Global Importance
    logger.info("Step 7a: Generating SHAP feature importance...")
    X_test = test_df[FEATURE_COLUMNS]
    global_importance = get_global_feature_importance(model, X_test)

    print("-" * 50)
    print(" SHAP GLOBAL FEATURE IMPORTANCE (Mean |SHAP|)")
    print("-" * 50)
    for idx, item in enumerate(global_importance, 1):
        print(f"  {idx:2d}. {item['feature']:<22}: {item['mean_abs_shap']:.5f}")
    print("-" * 50 + "\n")

    # Individual Prediction & SHAP Breakdown for the most recent trading day
    latest_row = test_df.iloc[-1]
    instance_explanation = explain_instance(model, latest_row, FEATURE_COLUMNS)
    latest_ticker = latest_row["ticker"]
    latest_date = str(latest_row["date"])
    pred_label = "BUY" if latest_row["prediction"] == 1 else "NOT BUY"
    conf = float(latest_row["confidence"])

    print("-" * 50)
    print(f" LATEST FORECAST FOR {latest_ticker} ({latest_date})")
    print("-" * 50)
    print(f"  Prediction  : {pred_label}")
    print(f"  Confidence  : {conf:.2%}")
    print("  Top Positive Drivers:")
    for d in instance_explanation["positive_drivers"][:3]:
        print(f"    + {d['feature']} = {d['value']} (SHAP impact: +{d['shap_impact']:.4f})")
    print("  Top Negative Drivers:")
    for d in instance_explanation["negative_drivers"][:3]:
        print(f"    - {d['feature']} = {d['value']} (SHAP impact: {d['shap_impact']:.4f})")
    print("-" * 50 + "\n")

    # Step 8: Quantitative Backtesting
    logger.info("Step 7b: Simulating quantitative strategy backtest...")
    backtest_results = run_backtest(test_df, threshold=confidence_thresh)

    print("=" * 80)
    print(" QUANTITATIVE BACKTEST SUMMARY (Strategy vs Buy & Hold)")
    print("=" * 80)
    for horizon, res in backtest_results.items():
        if "total_trades" in res and res["total_trades"] > 0:
            print(f" Horizon: {horizon.upper()}")
            print(f"   Trades Executed   : {res['total_trades']} (Win Rate: {res['win_rate']}%)")
            print(f"   Cumulative Return : Strategy={res['cumulative_strategy_return']}% | Benchmark={res['benchmark_cumulative_return']}%")
            print(f"   Sharpe Ratio      : {res['sharpe_ratio']}")
            print(f"   Maximum Drawdown  : {res['max_drawdown']}%\n")
    print("=" * 80 + "\n")

    # Persist Latest Prediction to Database
    db = SessionLocal()
    try:
        pred_record = Prediction(
            ticker=latest_ticker,
            prediction=pred_label,
            confidence=conf,
            model_version="xgboost_v1",
            shap_drivers=instance_explanation,
        )
        db.add(pred_record)
        db.commit()
        logger.info(f"Saved forecast for {latest_ticker} to predictions table.")
    except Exception as e:
        db.rollback()
        logger.warning(f"Failed to persist prediction record: {e}")
    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Financial Intelligence ML Pipeline Runner")
    parser.add_argument(
        "--tickers",
        type=str,
        default="NVDA,AAPL,MSFT",
        help="Comma-separated ticker list (e.g. NVDA,AAPL,MSFT)",
    )
    parser.add_argument(
        "--period",
        type=str,
        default="2y",
        help="yfinance market history period (e.g. 1y, 2y, 5y)",
    )
    parser.add_argument(
        "--skip-news",
        action="store_true",
        help="Skip fetching live news and use existing cached database articles",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
        help="BUY signal probability threshold for backtesting (default: 0.50)",
    )
    args = parser.parse_args()

    ticker_list = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    run_pipeline(
        tickers=ticker_list,
        period=args.period,
        skip_news=args.skip_news,
        confidence_thresh=args.threshold,
    )
