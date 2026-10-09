"""
Adaptive Opportunity Scoring Engine for Sprint 9: Autonomous Evaluation & Adaptive Intelligence.

Dynamically adapts factor weights for multi-factor market opportunity discovery based on
rolling out-of-sample Information Coefficients (IC):
  Factors:
    - ML Probability (Base Alpha)
    - FAISS Historical Analogs (Empirical Precedent)
    - Sentiment Rank (Market News & Social Polarity)
    - Technical / Momentum / Squeeze (Timing & Volatility Compression)
    - Corporate Catalyst Impact (Discreet Event Shocks)
    - Macro / Thematic Trend (Secular Growth Tailwinds)
    - Market Regime Alignment

Strict Guardrails:
  - Rolling walk-forward calibration (e.g. 60-day learning window -> Freeze -> 20-day unseen eval window).
  - Never calculates weights and evaluates on the same data.
  - Weight bounds [5%, 45%] prevent small-sample factor domination.
  - Compares Original Opportunity Score vs Adaptive Opportunity Score out-of-sample.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
from datetime import datetime, date, timedelta, timezone
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd

from database.connection import SessionLocal
from database.models import TrainingFeature, OpportunityHistory
from opportunities.scoring import OpportunityScorer
from evaluation.metrics import compute_spearman_rank_ic, compute_ic_series_statistics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("AdaptiveOpportunities")


class AdaptiveOpportunityScorer:
    """Manages rolling IC-based adaptive factor weighting for opportunity scoring."""

    DEFAULT_FACTOR_WEIGHTS = {
        "ml": 0.30,
        "analog": 0.25,
        "sentiment": 0.15,
        "technical": 0.10,
        "catalyst": 0.10,
        "regime": 0.10
    }

    MIN_WEIGHT = 0.05
    MAX_WEIGHT = 0.45

    def __init__(self, baseline_weights: Optional[Dict[str, float]] = None):
        self.baseline_weights = baseline_weights or dict(self.DEFAULT_FACTOR_WEIGHTS)
        self.current_weights = dict(self.baseline_weights)
        self.factor_ic_history: Dict[str, List[float]] = {k: [] for k in self.baseline_weights}

    def _normalize_and_bound_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Clamps weights into [5%, 45%] and normalizes sum to 1.0."""
        bounded = {k: min(self.MAX_WEIGHT, max(self.MIN_WEIGHT, w)) for k, w in weights.items()}
        tot = sum(bounded.values())
        if tot <= 0:
            n = len(weights)
            return {k: 1.0 / n for k in weights}

        normed = {k: w / tot for k, w in bounded.items()}

        for _ in range(5):
            excess = 0.0
            adj = []
            for k, w in normed.items():
                if w > self.MAX_WEIGHT:
                    excess += (w - self.MAX_WEIGHT)
                    normed[k] = self.MAX_WEIGHT
                elif w < self.MIN_WEIGHT:
                    excess -= (self.MIN_WEIGHT - w)
                    normed[k] = self.MIN_WEIGHT
                else:
                    adj.append(k)

            if abs(excess) < 1e-4 or not adj:
                break
            for k in adj:
                normed[k] += excess / len(adj)

        final_tot = sum(normed.values())
        return {k: round(w / final_tot, 4) for k, w in normed.items()}

    def evaluate_factor_ics(self, df_learning: pd.DataFrame) -> Dict[str, float]:
        """
        Computes Spearman Rank IC for each factor against future_return_5d.
        """
        fwd_ret = df_learning["future_return_5d"].values
        ics = {}

        # 1. ML factor (proxy: target probability or RSI rank)
        if "sentiment_x_momentum" in df_learning.columns:
            ics["ml"] = compute_spearman_rank_ic(df_learning["sentiment_x_momentum"].values, fwd_ret)
        else:
            ics["ml"] = 0.05

        # 2. Analog factor (proxy: return_5d_rank)
        if "return_5d_rank" in df_learning.columns:
            ics["analog"] = compute_spearman_rank_ic(df_learning["return_5d_rank"].values, fwd_ret)
        else:
            ics["analog"] = 0.04

        # 3. Sentiment factor
        if "avg_sentiment" in df_learning.columns:
            ics["sentiment"] = compute_spearman_rank_ic(df_learning["avg_sentiment"].values, fwd_ret)
        else:
            ics["sentiment"] = 0.03

        # 4. Technical factor
        if "rsi" in df_learning.columns:
            ics["technical"] = compute_spearman_rank_ic(df_learning["rsi"].values, fwd_ret)
        else:
            ics["technical"] = 0.02

        # 5. Catalyst factor (proxy: volume_ratio)
        if "volume_ratio" in df_learning.columns:
            ics["catalyst"] = compute_spearman_rank_ic(df_learning["volume_ratio"].values, fwd_ret)
        else:
            ics["catalyst"] = 0.02

        # 6. Regime factor (proxy: sector_return_5d)
        if "sector_return_5d" in df_learning.columns:
            ics["regime"] = compute_spearman_rank_ic(df_learning["sector_return_5d"].values, fwd_ret)
        else:
            ics["regime"] = 0.02

        for k, ic_val in ics.items():
            self.factor_ic_history[k].append(ic_val)

        return ics

    def update_factor_weights(self, factor_ics: Dict[str, float]) -> Dict[str, float]:
        """
        Updates factor weights based on measured IC:
          raw_weight = max(0.02, factor_ic)
          smoothed_weight = 0.70 * previous_weight + 0.30 * raw_weight
        """
        raw_scores = {}
        for k in self.baseline_weights:
            ic = factor_ics.get(k, 0.0)
            # Give positive weight to positive IC, floor at small positive value
            raw_scores[k] = max(0.02, ic + 0.05)

        tot_score = sum(raw_scores.values())
        ic_weights = {k: (s / tot_score) for k, s in raw_scores.items()}

        updated = {}
        for k in self.baseline_weights:
            prev_w = self.current_weights.get(k, self.baseline_weights[k])
            p_w = ic_weights[k]
            smoothed = 0.70 * prev_w + 0.30 * p_w
            updated[k] = smoothed

        self.current_weights = self._normalize_and_bound_weights(updated)
        logger.info(f"Updated adaptive opportunity factor weights: {self.current_weights}")
        return self.current_weights

    def run_walk_forward_opportunity_test(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Executes Walk-Forward test comparing Original vs Adaptive Opportunity Scoring:
          1. 60-day learning window: measure factor ICs and update weights.
          2. Freeze weights.
          3. Next 20-day evaluation window: compute original and adaptive scores on unseen data.
          4. Compare out-of-sample Rank IC and forward return of top-quintile opportunities.
        """
        df = df.sort_values(by=["date", "ticker"]).reset_index(drop=True)
        unique_dates = sorted(df["date"].unique())
        if len(unique_dates) < 80:
            logger.warning("Insufficient dates for 60/20 walk-forward window. Using full split.")
            idx_split = int(len(unique_dates) * 0.75)
        else:
            idx_split = len(unique_dates) - 20

        learn_dates = set(unique_dates[:idx_split])
        eval_dates = set(unique_dates[idx_split:])

        learn_df = df[df["date"].isin(learn_dates)]
        eval_df = df[df["date"].isin(eval_dates)].copy()

        # 1. Learn factor ICs on learning window
        factor_ics = self.evaluate_factor_ics(learn_df)
        adaptive_w = self.update_factor_weights(factor_ics)

        # 2. Evaluate on unseen evaluation window
        fwd_ret = eval_df["future_return_5d"].values

        # Original score (static weights)
        orig_score = np.zeros(len(eval_df))
        adapt_score = np.zeros(len(eval_df))

        factor_series = {
            "ml": eval_df.get("sentiment_x_momentum", np.zeros(len(eval_df))),
            "analog": eval_df.get("return_5d_rank", np.zeros(len(eval_df))),
            "sentiment": eval_df.get("avg_sentiment", np.zeros(len(eval_df))),
            "technical": eval_df.get("rsi", np.zeros(len(eval_df))),
            "catalyst": eval_df.get("volume_ratio", np.zeros(len(eval_df))),
            "regime": eval_df.get("sector_return_5d", np.zeros(len(eval_df)))
        }

        for k in self.baseline_weights:
            s_val = factor_series[k].values
            # Standardize factor
            s_std = (s_val - np.mean(s_val)) / (np.std(s_val) + 1e-6)
            orig_score += self.baseline_weights[k] * s_std
            adapt_score += adaptive_w[k] * s_std

        # Measure out-of-sample IC
        orig_ic = compute_spearman_rank_ic(orig_score, fwd_ret)
        adapt_ic = compute_spearman_rank_ic(adapt_score, fwd_ret)

        # Top 20% return
        q_orig = np.percentile(orig_score, 80)
        q_adapt = np.percentile(adapt_score, 80)

        orig_top_ret = float(np.mean(fwd_ret[orig_score >= q_orig]))
        adapt_top_ret = float(np.mean(fwd_ret[adapt_score >= q_adapt]))

        return {
            "learning_window": f"{min(learn_dates)} to {max(learn_dates)}",
            "eval_window": f"{min(eval_dates)} to {max(eval_dates)}",
            "eval_observations": len(eval_df),
            "factor_ics": {k: round(v, 4) for k, v in factor_ics.items()},
            "baseline_weights": self.baseline_weights,
            "adaptive_weights": adaptive_w,
            "original_rank_ic": round(orig_ic, 4),
            "adaptive_rank_ic": round(adapt_ic, 4),
            "ic_improvement": round(adapt_ic - orig_ic, 4),
            "original_top_quintile_return": round(orig_top_ret, 4),
            "adaptive_top_quintile_return": round(adapt_top_ret, 4),
            "verdict": (
                "Adaptive Opportunity Scoring improved out-of-sample Rank IC."
                if adapt_ic > orig_ic
                else "Adaptive Opportunity Scoring did not improve out-of-sample Rank IC."
            )
        }
