"""
DataStorm 2026 - Primary Latent Demand Heuristic (Option A)
==========================================================
Interpretable multiplicative model for maximum monthly potential under
left-censored (supply-capped) observed sales.

Production submissions use this module. ML quantile models are benchmarks only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Uplift weights (documented for judges / paper)
CENSORING_UPLIFT = 1.5000
CATCHMENT_UPLIFT = 0.2466
HIGH_CENSORING_THRESHOLD = 0.40
HIGH_CENSORING_CALIBRATION_SLOPE = 0.33
MAX_POTENTIAL_MULTIPLIER = 5.0
MIN_PREDICTION_LITERS = 1.0
MIN_FLOOR_VS_MEDIAN_RATIO = 0.05

PRIMARY_MODEL_NAME = "Heuristic_Latent"


def ensure_heuristic_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """Fill columns required by the latent heuristic (safe for validation folds)."""
    out = df.copy()
    defaults = {
        "jan_base": out.get("hist_median_vol", pd.Series(0.0, index=out.index)),
        "size_factor": 1.0,
        "type_factor": 1.0,
        "target_season_factor": 1.0,
        "censoring_score": 0.0,
        "peer_efficiency_gap": 1.0,
        "combined_catchment_score": 0.0,
        "competition_dampener": 1.0,
        "hist_median_vol": 0.0,
        "hist_max_vol": 0.0,
    }
    for col, default in defaults.items():
        if col not in out.columns:
            if isinstance(default, pd.Series):
                out[col] = default.reindex(out.index).fillna(0.0)
            else:
                out[col] = default
        else:
            if col in ("competition_dampener", "peer_efficiency_gap", "size_factor", "type_factor", "target_season_factor"):
                out[col] = out[col].fillna(default)
            else:
                out[col] = pd.to_numeric(out[col], errors="coerce").fillna(
                    0.0 if col != "competition_dampener" else default
                )
    if "competition_dampener" in out.columns:
        out["competition_dampener"] = out["competition_dampener"].fillna(1.0).clip(0.5, 1.0)
    if "peer_efficiency_gap" in out.columns:
        out["peer_efficiency_gap"] = out["peer_efficiency_gap"].fillna(1.0).clip(1.0, 3.0)
    if "combined_catchment_score" in out.columns:
        out["combined_catchment_score"] = out["combined_catchment_score"].fillna(0.0).clip(0.0, 1.0)
    if "censoring_score" in out.columns:
        out["censoring_score"] = out["censoring_score"].fillna(0.0).clip(0.0, 1.0)
    return out


def compute_heuristic_potential(df: pd.DataFrame) -> np.ndarray:
    """Core latent-demand formula (multiplicative uncapping)."""
    d = ensure_heuristic_inputs(df)
    return (
        d["jan_base"].values
        * d["size_factor"].values
        * d["type_factor"].values
        * d["target_season_factor"].values
        * (1.0 + d["censoring_score"].values * CENSORING_UPLIFT)
        * d["peer_efficiency_gap"].values
        * (1.0 + d["combined_catchment_score"].values * CATCHMENT_UPLIFT)
        * d["competition_dampener"].values
    )


def finalize_latent_predictions(raw: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    """Apply business floors, hard ceiling vs history, and high-censoring calibration."""
    d = ensure_heuristic_inputs(df)
    pred = np.asarray(raw, dtype=float).copy()

    # Floor: no zero-potential outlets with any history
    median = d["hist_median_vol"].values
    floor = np.maximum(MIN_PREDICTION_LITERS, median * MIN_FLOOR_VS_MEDIAN_RATIO)
    pred = np.maximum(pred, floor)

    # Hard cap vs historical baseline (latent cannot exceed 5x jan_base)
    base = np.maximum(d["jan_base"].values, median)
    cap = base * MAX_POTENTIAL_MULTIPLIER
    pred = np.minimum(pred, cap)

    # Additional uplift for strongly supply-capped outlets
    cens = d["censoring_score"].values
    high = cens > HIGH_CENSORING_THRESHOLD
    if np.any(high):
        uplift = 1.0 + (cens[high] - HIGH_CENSORING_THRESHOLD) * HIGH_CENSORING_CALIBRATION_SLOPE
        pred[high] = pred[high] * uplift
        pred = np.minimum(pred, cap)

    return np.round(pred, 2)
