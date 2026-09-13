import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
from typing import Dict, Any
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("EDA")


def run_eda(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Executes exploratory data analysis and data quality verification
    on the engineered training dataset.
    """
    print("\n" + "=" * 70)
    print(" EXPLORATORY DATA ANALYSIS (EDA) & QUALITY CHECKS")
    print("=" * 70)

    # 1. Dataset Shape and Uniqueness
    total_rows = len(df)
    unique_tickers = df["ticker"].nunique()
    has_duplicates = bool(df.duplicated(subset=["ticker", "date"]).any())

    print(f"\n1. DATASET INTEGRITY:")
    print(f"   Total Observations   : {total_rows}")
    print(f"   Unique Tickers       : {unique_tickers} ({', '.join(df['ticker'].unique())})")
    print(f"   Duplicate (ticker,date) : {'DETECTED (FAIL)' if has_duplicates else 'NONE (PASS)'}")

    # 2. Missing Value Check
    print(f"\n2. MISSING VALUES CHECK:")
    null_counts = df.isnull().sum()
    features_with_nulls = null_counts[null_counts > 0]
    if features_with_nulls.empty:
        print("   Zero missing values in engineered dataset! (PASS)")
    else:
        for col, count in features_with_nulls.items():
            pct = (count / total_rows) * 100
            print(f"   - {col:<22}: {count} nulls ({pct:.2f}%)")

    # 3. Target Distribution Check (Class Imbalance)
    print(f"\n3. TARGET DISTRIBUTION (Class Balance):")
    target_counts = df["target"].value_counts().to_dict()
    buy_count = target_counts.get(1, 0)
    not_buy_count = target_counts.get(0, 0)
    buy_pct = (buy_count / total_rows) * 100 if total_rows > 0 else 0
    not_buy_pct = (not_buy_count / total_rows) * 100 if total_rows > 0 else 0

    print(f"   Class 1 (BUY / Return > 0)    : {buy_count:5d} ({buy_pct:.2f}%)")
    print(f"   Class 0 (NOT BUY / Return <= 0): {not_buy_count:5d} ({not_buy_pct:.2f}%)")
    target_all_valid = bool(df["target"].notnull().all())
    print(f"   Target Null Check             : {'PASS' if target_all_valid else 'FAIL'}")

    # 4. Correlation Analysis
    print(f"\n4. FEATURE CORRELATIONS WITH 5-DAY FUTURE RETURN:")
    correlations = {}
    target_col = "future_return_5d"
    features_to_correlate = [
        "avg_sentiment",
        "total_article_count",
        "sentiment_ema_3",
        "sentiment_delta",
        "rsi",
        "macd",
        "macd_diff",
        "ema20",
        "ema50",
        "volume_ratio",
        "atr",
        "return_5d",
        "spy_return_5d",
    ]

    for feat in features_to_correlate:
        if feat in df.columns and target_col in df.columns:
            valid_subset = df[[feat, target_col]].dropna()
            if len(valid_subset) > 1 and valid_subset[feat].std() > 1e-6 and valid_subset[target_col].std() > 1e-6:
                corr_val = float(valid_subset[feat].corr(valid_subset[target_col]))
                correlations[feat] = round(corr_val, 4)
                print(f"   corr({feat:<20}, future_return_5d) = {corr_val:+.4f}")
            elif feat.startswith("sentiment") or feat.startswith("avg") or "article" in feat:
                # Calculate correlation on days with actual published news
                news_active = df[df["total_article_count"] > 0][[feat, target_col]].dropna()
                if len(news_active) > 1 and news_active[feat].std() > 1e-6 and news_active[target_col].std() > 1e-6:
                    corr_val = float(news_active[feat].corr(news_active[target_col]))
                    correlations[feat] = round(corr_val, 4)
                    print(f"   corr({feat:<20}, future_return_5d) = {corr_val:+.4f} (on active news days, n={len(news_active)})")
                else:
                    print(f"   corr({feat:<20}, future_return_5d) = N/A (zero variance in window)")
            else:
                print(f"   corr({feat:<20}, future_return_5d) = N/A (zero variance)")

    print("=" * 70 + "\n")

    return {
        "total_rows": total_rows,
        "unique_tickers": unique_tickers,
        "has_duplicates": has_duplicates,
        "target_all_valid": target_all_valid,
        "class_balance": {"buy": buy_count, "not_buy": not_buy_count, "buy_pct": round(buy_pct, 2)},
        "correlations": correlations,
    }


if __name__ == "__main__":
    csv_path = Path(__file__).resolve().parent.parent / "data" / "training_dataset.csv"
    if csv_path.exists():
        df_loaded = pd.read_csv(csv_path)
        run_eda(df_loaded)
    else:
        print(f"Training dataset not found at {csv_path}. Run build_training_dataset.py first.")
