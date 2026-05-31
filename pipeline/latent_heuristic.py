"""
DataStorm 2026 - Primary Latent Demand Heuristic (Option A)
==========================================================
Two-regime interpretable model for maximum monthly potential under
left-censored (supply-capped) observed sales.

  - Low censoring  -> conservative baseline (size/type/season only)
  - High censoring -> full latent uncapping (peer gap + catchment + censoring uplift)

Production submissions use this module. ML quantile models are benchmarks only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from poi_catchment import enrich_catchment_features

# Uplift weights — recalibrate via pipeline/calibrate_heuristic.py (walk-forward ceiling targets)
CENSORING_UPLIFT = 1.0337
CATCHMENT_UPLIFT = 0.9769

# Two-regime blend: censoring_score in [0, BLEND_START] -> baseline; >= BLEND_FULL -> full latent
REGIME_BLEND_START = 0.15
REGIME_BLEND_FULL = 0.45

HIGH_CENSORING_THRESHOLD = 0.40
HIGH_CENSORING_CALIBRATION_SLOPE = 0.33
MAX_POTENTIAL_MULTIPLIER = 5.0
MIN_PREDICTION_LITERS = 1.0
MIN_FLOOR_VS_MEDIAN_RATIO = 0.05

PRIMARY_MODEL_NAME = "Heuristic_Latent_TwoRegime"


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
    if "censoring_score" in out.columns:
        out["censoring_score"] = out["censoring_score"].fillna(0.0).clip(0.0, 1.0)
    out = enrich_catchment_features(out)
    return out


def censoring_regime_weight(censoring: np.ndarray) -> np.ndarray:
    """Smooth blend weight w in [0, 1]: 0 = baseline, 1 = full latent uncapping."""
    span = REGIME_BLEND_FULL - REGIME_BLEND_START
    t = (np.asarray(censoring, dtype=float) - REGIME_BLEND_START) / (span + 1e-9)
    return np.clip(t, 0.0, 1.0)


def compute_baseline_potential(df: pd.DataFrame) -> np.ndarray:
    """Conservative potential for unconstrained outlets (no uncapping terms)."""
    d = ensure_heuristic_inputs(df)
    return (
        d["jan_base"].values
        * d["size_factor"].values
        * d["type_factor"].values
        * d["target_season_factor"].values
        * d["competition_dampener"].values
    )


def compute_latent_core(
    df: pd.DataFrame,
    censoring_uplift: float | None = None,
    catchment_uplift: float | None = None,
) -> np.ndarray:
    """Full latent uncapping formula (supply-cap aware)."""
    d = ensure_heuristic_inputs(df)
    alpha = CENSORING_UPLIFT if censoring_uplift is None else censoring_uplift
    gamma = CATCHMENT_UPLIFT if catchment_uplift is None else catchment_uplift
    return (
        d["jan_base"].values
        * d["size_factor"].values
        * d["type_factor"].values
        * d["target_season_factor"].values
        * (1.0 + d["censoring_score"].values * alpha)
        * d["peer_efficiency_gap"].values
        * (1.0 + d["effective_catchment_score"].values * gamma)
        * d["competition_dampener"].values
    )


def compute_heuristic_potential(
    df: pd.DataFrame,
    censoring_uplift: float | None = None,
    catchment_uplift: float | None = None,
) -> np.ndarray:
    """Two-regime blend: (1-w)*baseline + w*latent_core before finalize."""
    d = ensure_heuristic_inputs(df)
    baseline = compute_baseline_potential(d)
    latent = compute_latent_core(d, censoring_uplift, catchment_uplift)
    w = censoring_regime_weight(d["censoring_score"].values)
    return (1.0 - w) * baseline + w * latent


def finalize_latent_predictions(raw: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    """Apply business floors, hard ceiling vs history, and high-censoring calibration."""
    d = ensure_heuristic_inputs(df)
    pred = np.asarray(raw, dtype=float).copy()

    median = d["hist_median_vol"].values
    floor = np.maximum(MIN_PREDICTION_LITERS, median * MIN_FLOOR_VS_MEDIAN_RATIO)
    pred = np.maximum(pred, floor)

    base = np.maximum(d["jan_base"].values, median)
    cap = base * MAX_POTENTIAL_MULTIPLIER
    pred = np.minimum(pred, cap)

    cens = d["censoring_score"].values
    high = cens > HIGH_CENSORING_THRESHOLD
    if np.any(high):
        uplift = 1.0 + (cens[high] - HIGH_CENSORING_THRESHOLD) * HIGH_CENSORING_CALIBRATION_SLOPE
        pred[high] = pred[high] * uplift
        pred = np.minimum(pred, cap)

    return np.round(pred, 2)


def predict_latent_potential(
    df: pd.DataFrame,
    censoring_uplift: float | None = None,
    catchment_uplift: float | None = None,
) -> np.ndarray:
    """End-to-end production prediction (blend + finalize)."""
    raw = compute_heuristic_potential(df, censoring_uplift, catchment_uplift)
    return finalize_latent_predictions(raw, df)
