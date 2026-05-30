"""
DataStorm 2026 - Ceiling Validation Metrics
===========================================
Metrics for evaluating latent *potential* models against ceiling proxies,
not just next-month observed sales.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def ceiling_proxy(df: pd.DataFrame) -> np.ndarray:
    """Soft ceiling reference: max of p90 history and observed max."""
    p90 = df["hist_p90_vol"].values if "hist_p90_vol" in df.columns else df["hist_median_vol"].values
    hmax = df["hist_max_vol"].values if "hist_max_vol" in df.columns else p90
    return np.maximum(p90, hmax)


def compute_ceiling_metrics(
    y_pred: np.ndarray,
    df: pd.DataFrame,
    y_obs: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute ceiling-focused validation metrics for one prediction vector."""
    y_pred = np.asarray(y_pred, dtype=float)
    ceiling = ceiling_proxy(df)
    cens = df["censoring_score"].values if "censoring_score" in df.columns else np.zeros(len(df))

    rank_ceiling, _ = spearmanr(y_pred, ceiling)
    rank_p90, _ = spearmanr(y_pred, df["hist_p90_vol"].values if "hist_p90_vol" in df.columns else ceiling)

    log_err = np.abs(np.log1p(np.clip(y_pred, 0, None)) - np.log1p(np.clip(ceiling, 0, None)))
    w = 1.0 + 2.0 * cens
    metrics: Dict[str, Any] = {
        "rank_corr_ceiling_spearman": float(rank_ceiling) if not np.isnan(rank_ceiling) else 0.0,
        "rank_corr_p90_spearman": float(rank_p90) if not np.isnan(rank_p90) else 0.0,
        "log_mae_to_ceiling": float(np.mean(log_err)),
        "log_mae_to_ceiling_weighted": float(np.average(log_err, weights=w)),
    }

    high = cens > 0.40
    if np.sum(high) >= 30:
        metrics["log_mae_to_ceiling_high_cens"] = float(np.mean(log_err[high]))
        rc_hi, _ = spearmanr(y_pred[high], ceiling[high])
        metrics["rank_corr_ceiling_high_cens"] = float(rc_hi) if not np.isnan(rc_hi) else 0.0
    else:
        metrics["log_mae_to_ceiling_high_cens"] = float("nan")
        metrics["rank_corr_ceiling_high_cens"] = float("nan")

    if y_obs is not None:
        y_obs = np.asarray(y_obs, dtype=float)
        uplift = (y_pred - y_obs) / (y_obs + 1e-9)
        uc, _ = spearmanr(cens, uplift)
        metrics["uplift_censoring_spearman"] = float(uc) if not np.isnan(uc) else 0.0
        metrics["pct_pred_exceeds_observed"] = float(np.mean(y_pred > y_obs) * 100.0)
        if np.sum(high) >= 30:
            metrics["pct_exceeds_observed_high_cens"] = float(np.mean(y_pred[high] > y_obs[high]) * 100.0)
            metrics["mean_uplift_high_cens"] = float(np.mean(uplift[high]) * 100.0)
        else:
            metrics["pct_exceeds_observed_high_cens"] = float("nan")
            metrics["mean_uplift_high_cens"] = float("nan")

    return metrics
