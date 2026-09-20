# Equity Research Note: XOM

**Generated**: 2026-09-15 21:24 UTC  
**Primary Recommendation**: **NOT BUY**  
**Calibrated Probability**: **48.9%** (Decision Threshold: `60.0%`)  
**Model Confidence**: **51.0%**  
**Market Regime**: **Neutral** (VIX: `15.72`)  
**Risk Level**: **LOW** (Score: `0.30`)

---

## 1. Executive Summary & Investment Thesis

XOM maintains a **NOT BUY** stance from the quantitative ML model. Calibrated directional probability of **48.9%** falls short of the 60.0% conviction threshold, warranting capital preservation and defensive risk-budgeting. Macro regime is currently classified as **Neutral** (VIX at 15.7). From a cross-sectional perspective, XOM's FinBERT sentiment ranks at the 52% percentile of the 25-asset coverage universe, while 5-day sector peer momentum stands at +1.23% (SQUEEZE_ACTIVE). Primary model feature attributions (TreeSHAP) indicate key quantitative drivers: spy_return_1d (+0.002), macd_diff (+0.001), atr (+0.001), return_5d (+0.001). Historical analog matching across 30,025 situations identified 50 closely aligned multi-factor setups. Precedents achieved a **42.0% 5-day win rate**, delivering a median return of **-0.79%** (mean -0.65%, 95% Confidence Interval: [-1.80%, +0.50%]). Knowledge Graph multi-hop analysis traces structural impact: Macro / Sector Catalyst: Energy affects XOM (Energy sector earnings revisions directly elevate Exxon Mobil free cash flow.). Risk governance profile is LOW, monitoring: Benign macro regime and balanced momentum oscillators.

---

## 2. Quantitative ML Model Signal

- **Signal**: `NOT BUY`
- **Probability of Positive 5-Day Return**: `48.95%`
- **Decision Threshold (Sharpe-Optimized)**: `60.00%`
- **Edge Above Hurdle**: `-11.05%`
- **Underlying Engine**: `Random Forest (43 engineered features, 25-asset cross-sectional universe)`

---

## 3. SHAP Feature Attribution Analysis

Primary feature contributions identified via TreeSHAP explainability:

### Positive Tailwinds:
- **Positive**: `spy_return_1d (+0.002)`
- **Positive**: `macd_diff (+0.001)`
- **Positive**: `atr (+0.001)`

### Negative Headwinds:
- **Headwind**: `vix_level (-0.004)`
- **Headwind**: `bollinger_pband (-0.003)`
- **Headwind**: `bb_squeeze (-0.003)`

---

## 4. Historical Analog Pattern Matching (FAISS 30k Engine)

Vector similarity search queried against **30,025 historical observations** across 25 tickers (2021–2026):

- **Sample Size**: `50 setups`
- **Historical 5-Day Win Rate**: `42.0%`
- **Median Forward Return**: `-0.79%`
- **Average Forward Return**: `-0.65%`
- **95% Confidence Interval**: `[-1.80%, +0.50%]`
- **Regime Distribution**: `{'Neutral': 0.56, 'Bull': 0.22, 'Low Volatility': 0.22}`
- **Sector Distribution**: `{'Consumer Staples': 0.12, 'Energy': 0.2, 'Healthcare': 0.08, 'Industrials': 0.04, 'Technology': 0.28, 'Financials': 0.14, 'Consumer Discretionary': 0.12, 'Communication Services': 0.02}`

### Nearest Historical Precedents:
- `2025-11-12` (WMT, Consumer Staples): 5d Return: `-2.74%` | Regime: *Neutral* | Event: *neutral_regime* (Dist: 0.682)
- `2026-09-08` (XOM, Energy): 5d Return: `+5.00%` | Regime: *Neutral* | Event: *market_catalyst* (Dist: 0.9129)
- `2026-04-28` (WMT, Consumer Staples): 5d Return: `+2.51%` | Regime: *Bull* | Event: *neutral_regime* (Dist: 0.9584)
- `2026-09-08` (PFE, Healthcare): 5d Return: `-0.95%` | Regime: *Neutral* | Event: *market_catalyst* (Dist: 0.9951)


---

## 5. Technical Indicator & Momentum Setup

- **RSI (14)**: `53.7` (NEUTRAL) — **Cross-Sectional Rank**: `60%`
- **MACD Structure**: `BEARISH_DIVERGENCE` (Diff: `-0.600`)
- **Trend Orientation**: `BULLISH_UPTREND` (EMA20: `159.95` vs EMA50: `155.51`)
- **Bollinger Squeeze**: `SQUEEZE_ACTIVE` (Score: `0.67`)
- **Volume Ratio**: `0.92x` (NORMAL)

---

## 6. Market Regime & Cross-Sectional Ranking

- **Current Regime**: `Neutral`
- **VIX Level**: `15.72`
- **Sentiment Cross-Sectional Rank**: `52%` (vs 25 stocks in coverage)
- **Sector 5-Day Momentum**: `+1.23%`

---

## 7. Key Risks & Invalidation Triggers

- Benign macro regime and balanced momentum oscillators


---

## 8. Multi-Hop Knowledge Graph Cascades

- Macro / Sector Catalyst: Energy affects XOM (Energy sector earnings revisions directly elevate Exxon Mobil free cash flow.)
- Sector Peer Correlation: XOM anchored within Energy (Exxon Mobil explores, produces, and refines crude oil and petrochemicals.)


---

*Report synthesized autonomously by the AI Financial Research Assistant (Sprint 4.1 Architecture).*
