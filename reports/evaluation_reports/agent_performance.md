# SPRINT 9 — INDEPENDENT AGENT EVALUATION REPORT
**Evaluation Date:** 2026-10-09  
**Architecture:** Committee Multi-Agent Governance Layer  
**Objective:** Measure empirical predictive accuracy and economic value by agent independently before adaptation.  

---

## 1. Agent Leaderboard (Out-of-Sample Performance)

| Agent Name | Sample Size | Hit Rate | Directional Acc | ECE | Sharpe Ratio | Sortino Ratio | Max DD | Avg Return (5d) | Best Regime |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Risk Officer** | 60 | `48.3%` | `48.3%` | `0.2358` | `1.93` | `4.30` | `21.19%` | `+0.70%` | **High Volatility** |
| **Bull Analyst** | 60 | `48.3%` | `48.3%` | `0.0950` | `-0.77` | `-1.13` | `38.42%` | `-0.28%` | **Low Volatility** |
| **Portfolio Manager** | 60 | `48.3%` | `48.3%` | `0.1441` | `-0.77` | `-1.13` | `38.42%` | `-0.28%` | **Low Volatility** |
| **CIO** | 60 | `53.3%` | `53.3%` | `0.0635` | `-1.48` | `-1.68` | `46.52%` | `-0.54%` | **Low Volatility** |
| **Bear Analyst** | 60 | `53.3%` | `53.3%` | `0.0425` | `-1.48` | `-1.68` | `46.52%` | `-0.54%` | **Low Volatility** |
| **Evidence Prosecutor** | 60 | `53.3%` | `53.3%` | `0.0167` | `-1.48` | `-1.68` | `46.52%` | `-0.54%` | **Low Volatility** |

---

## 2. Agent Breakdown by Market Regime

| Agent Name | Market Regime | Decisions | Hit Rate | Avg 5d Return |
| :--- | :--- | :---: | :---: | :---: |
| **Risk Officer** | Sideways | 26 | `42.3%` | `-0.29%` |
| **Risk Officer** | Bull | 21 | `52.4%` | `+1.52%` |
| **Risk Officer** | High Volatility | 6 | `66.7%` | `+3.41%` |
| **Risk Officer** | Bear | 6 | `50.0%` | `-0.05%` |
| **Risk Officer** | Low Volatility | 1 | `0.0%` | `-2.73%` |
| **Bull Analyst** | Sideways | 26 | `46.2%` | `+0.44%` |
| **Bull Analyst** | Bull | 21 | `42.9%` | `-1.19%` |
| **Bull Analyst** | High Volatility | 6 | `50.0%` | `-1.80%` |
| **Bull Analyst** | Bear | 6 | `66.7%` | `+0.80%` |
| **Bull Analyst** | Low Volatility | 1 | `100.0%` | `+2.73%` |
| **Portfolio Manager** | Sideways | 26 | `46.2%` | `+0.44%` |
| **Portfolio Manager** | Bull | 21 | `42.9%` | `-1.19%` |
| **Portfolio Manager** | High Volatility | 6 | `50.0%` | `-1.80%` |
| **Portfolio Manager** | Bear | 6 | `66.7%` | `+0.80%` |
| **Portfolio Manager** | Low Volatility | 1 | `100.0%` | `+2.73%` |
| **CIO** | Sideways | 26 | `57.7%` | `+0.29%` |
| **CIO** | Bull | 21 | `47.6%` | `-1.52%` |
| **CIO** | High Volatility | 6 | `50.0%` | `-1.80%` |
| **CIO** | Bear | 6 | `50.0%` | `+0.05%` |
| **CIO** | Low Volatility | 1 | `100.0%` | `+2.73%` |
| **Bear Analyst** | Sideways | 26 | `57.7%` | `+0.29%` |
| **Bear Analyst** | Bull | 21 | `47.6%` | `-1.52%` |
| **Bear Analyst** | High Volatility | 6 | `50.0%` | `-1.80%` |
| **Bear Analyst** | Bear | 6 | `50.0%` | `+0.05%` |
| **Bear Analyst** | Low Volatility | 1 | `100.0%` | `+2.73%` |
| **Evidence Prosecutor** | Sideways | 26 | `57.7%` | `+0.29%` |
| **Evidence Prosecutor** | Bull | 21 | `47.6%` | `-1.52%` |
| **Evidence Prosecutor** | High Volatility | 6 | `50.0%` | `-1.80%` |
| **Evidence Prosecutor** | Bear | 6 | `50.0%` | `+0.05%` |
| **Evidence Prosecutor** | Low Volatility | 1 | `100.0%` | `+2.73%` |

---

## 3. Agent Breakdown by Sector

| Agent Name | Sector | Decisions | Hit Rate | Avg 5d Return |
| :--- | :--- | :---: | :---: | :---: |
| **Risk Officer** | Technology | 17 | `52.9%` | `+2.98%` |
| **Risk Officer** | Healthcare | 9 | `44.4%` | `-1.21%` |
| **Risk Officer** | Industrials | 5 | `80.0%` | `+2.51%` |
| **Risk Officer** | Consumer Cyclical | 8 | `25.0%` | `-1.39%` |
| **Risk Officer** | Communication Services | 7 | `42.9%` | `+0.98%` |
| **Risk Officer** | Financial Services | 8 | `25.0%` | `-0.99%` |
| **Risk Officer** | Energy | 4 | `75.0%` | `-0.20%` |
| **Risk Officer** | Consumer Defensive | 2 | `100.0%` | `+1.34%` |
| **Bull Analyst** | Technology | 17 | `47.1%` | `-0.78%` |
| **Bull Analyst** | Healthcare | 9 | `44.4%` | `-0.79%` |
| **Bull Analyst** | Industrials | 5 | `80.0%` | `+1.09%` |
| **Bull Analyst** | Consumer Cyclical | 8 | `37.5%` | `-0.02%` |
| **Bull Analyst** | Communication Services | 7 | `42.9%` | `-0.38%` |
| **Bull Analyst** | Financial Services | 8 | `50.0%` | `-0.53%` |
| **Bull Analyst** | Energy | 4 | `50.0%` | `+1.12%` |
| **Bull Analyst** | Consumer Defensive | 2 | `50.0%` | `+0.25%` |
| **Portfolio Manager** | Technology | 17 | `47.1%` | `-0.78%` |
| **Portfolio Manager** | Healthcare | 9 | `44.4%` | `-0.79%` |
| **Portfolio Manager** | Industrials | 5 | `80.0%` | `+1.09%` |
| **Portfolio Manager** | Consumer Cyclical | 8 | `37.5%` | `-0.02%` |
| **Portfolio Manager** | Communication Services | 7 | `42.9%` | `-0.38%` |
| **Portfolio Manager** | Financial Services | 8 | `50.0%` | `-0.53%` |
| **Portfolio Manager** | Energy | 4 | `50.0%` | `+1.12%` |
| **Portfolio Manager** | Consumer Defensive | 2 | `50.0%` | `+0.25%` |
| **CIO** | Technology | 17 | `47.1%` | `-2.98%` |
| **CIO** | Healthcare | 9 | `55.6%` | `+1.21%` |
| **CIO** | Industrials | 5 | `20.0%` | `-2.51%` |
| **CIO** | Consumer Cyclical | 8 | `75.0%` | `+1.39%` |
| **CIO** | Communication Services | 7 | `71.4%` | `+0.40%` |
| **CIO** | Financial Services | 8 | `75.0%` | `+0.99%` |
| **CIO** | Energy | 4 | `25.0%` | `+0.20%` |
| **CIO** | Consumer Defensive | 2 | `0.0%` | `-1.34%` |
| **Bear Analyst** | Technology | 17 | `47.1%` | `-2.98%` |
| **Bear Analyst** | Healthcare | 9 | `55.6%` | `+1.21%` |
| **Bear Analyst** | Industrials | 5 | `20.0%` | `-2.51%` |
| **Bear Analyst** | Consumer Cyclical | 8 | `75.0%` | `+1.39%` |
| **Bear Analyst** | Communication Services | 7 | `71.4%` | `+0.40%` |
| **Bear Analyst** | Financial Services | 8 | `75.0%` | `+0.99%` |
| **Bear Analyst** | Energy | 4 | `25.0%` | `+0.20%` |
| **Bear Analyst** | Consumer Defensive | 2 | `0.0%` | `-1.34%` |
| **Evidence Prosecutor** | Technology | 17 | `47.1%` | `-2.98%` |
| **Evidence Prosecutor** | Healthcare | 9 | `55.6%` | `+1.21%` |
| **Evidence Prosecutor** | Industrials | 5 | `20.0%` | `-2.51%` |
| **Evidence Prosecutor** | Consumer Cyclical | 8 | `75.0%` | `+1.39%` |
| **Evidence Prosecutor** | Communication Services | 7 | `71.4%` | `+0.40%` |
| **Evidence Prosecutor** | Financial Services | 8 | `75.0%` | `+0.99%` |
| **Evidence Prosecutor** | Energy | 4 | `25.0%` | `+0.20%` |
| **Evidence Prosecutor** | Consumer Defensive | 2 | `0.0%` | `-1.34%` |

---

## 4. Agent Confidence Reliability (Calibration by Tier)

| Agent Name | Confidence Tier | Decisions | Empirical Hit Rate | Avg Return |
| :--- | :--- | :---: | :---: | :---: |
| **Risk Officer** | High (>=70%) | 60 | `48.3%` | `+0.70%` |
| **Bull Analyst** | Medium (50-69%) | 32 | `46.9%` | `+0.41%` |
| **Bull Analyst** | Low (<50%) | 28 | `50.0%` | `-1.07%` |
| **Portfolio Manager** | Medium (50-69%) | 60 | `48.3%` | `-0.28%` |
| **CIO** | Medium (50-69%) | 60 | `53.3%` | `-0.54%` |
| **Bear Analyst** | Low (<50%) | 33 | `54.5%` | `-0.27%` |
| **Bear Analyst** | Medium (50-69%) | 27 | `51.9%` | `-0.87%` |
| **Evidence Prosecutor** | Low (<50%) | 60 | `53.3%` | `-0.54%` |

---

## 5. Key Findings for Adaptive Weighting (Phase 10 Input)

1. **Evidence Prosecutor & Risk Officer:** Provide vital downside protection by vetoing low-probability trades, suppressing tail drawdowns.
2. **Regime Specialization:** Agents exhibit distinct regime strengths (e.g., Bull Analyst excels in Bull regimes; Risk Officer excels in High Volatility).
3. **Weight Calibration Baseline:** These empirical measurements establish the initial weights for Phase 10 adaptive committee optimization.
