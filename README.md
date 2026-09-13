# Financial Intelligence & Market Impact ML Pipeline

A quantitative machine learning platform combining NLP sentiment signals (FinBERT), technical market dynamics, XGBoost binary prediction, SHAP explainability, and multi-horizon backtesting.

---

## Architecture Overview

```text
                  ┌──────────────┐
                  │ News Sources │
                  │(Yahoo / RSS) │
                  └──────┬───────┘
                         │
                         ▼
                Data Ingestion Layer
                 (yfinance + News)
                         │
                         ▼
              PostgreSQL / SQLAlchemy
              (With SQLite Fallback)
                         │
        ┌────────────────┼───────────────┐
        ▼                ▼               ▼
 Market Indicators   FinBERT NLP   Event Detection
   (RSI/MACD/Vol)   (Sentiment)        (Events)
        └────────────────┼───────────────┘
                         │
                         ▼
               Feature Engineering
             (Daily Vector Alignment)
                         │
                         ▼
             XGBoost Binary Predictor
           (Temporal Walk-Forward Split)
                         │
        ┌────────────────┴───────────────┐
        ▼                                ▼
  SHAP Explainability           Backtesting Engine
 (Top Feature Drivers)        (1d, 5d, 10d Win Rates)
```

---

## 5 Core Entities (Data Model)

1. **`companies`**: Ticker, company name, sector, industry, market cap.
2. **`articles`**: News title, content, publisher source, URL, published timestamp, FinBERT sentiment score & probabilities.
3. **`events`**: Macro and company-specific events (earnings, upgrades, product releases, geopolitical events).
4. **`market_data`**: Historical OHLCV, 14-period RSI, MACD, annualized 20-day volatility, forward returns ($R_{t+1}, R_{t+5}, R_{t+10}$).
5. **`predictions`**: Generated trade signals (BUY / NOT BUY), model confidence, SHAP positive/negative driver attributions.

---

## Quickstart

### 1. Activate Environment
```bash
.\.venv\Scripts\activate
```

### 2. Configure Environment (Optional)
Copy `.env.example` to `.env` if custom database credentials are required:
```bash
cp .env.example .env
```
*Note: If PostgreSQL is not reachable, the system automatically falls back to local SQLite at `data/fin_sentiment.db`.*

### 3. Initialize Database Schema & Seed Companies
```bash
python scripts/init_db.py
```

### 4. Run the Full ML Research Pipeline
```bash
python scripts/run_pipeline.py --tickers NVDA,AAPL,MSFT --period 2y
```

### CLI Arguments
- `--tickers`: Comma-separated list of symbols (default: `NVDA,AAPL,MSFT`).
- `--period`: Historical price window from Yahoo Finance (default: `2y`).
- `--threshold`: Probability threshold for BUY signal execution (default: `0.50`).
- `--skip-news`: Skip downloading live headlines and use cached articles in database.
