"""
DataStorm 2026 - Exact Heuristic Component Attribution
======================================================
Waterfall-style decomposition of the two-regime latent heuristic.
Latent-only terms (censoring, peer, catchment) are scaled by regime weight w.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from latent_heuristic import (
    CATCHMENT_UPLIFT,
    CENSORING_UPLIFT,
    censoring_regime_weight,
    compute_baseline_potential,
    compute_heuristic_potential,
    ensure_heuristic_inputs,
    finalize_latent_predictions,
)

COMPONENT_NAMES = [
    "Size Factor",
    "Type Factor",
    "Seasonality",
    "Competition Dampener",
    "Censoring Uplift (regime-weighted)",
    "Peer Efficiency Gap (regime-weighted)",
    "Catchment Uplift (regime-weighted)",
]


def compute_heuristic_attributions(df: pd.DataFrame) -> dict:
    """Return per-outlet log-space waterfall attributions for the two-regime heuristic."""
    d = ensure_heuristic_inputs(df)
    n = len(d)
    w = censoring_regime_weight(d["censoring_score"].values)

    shared_terms = [
        d["size_factor"].values.astype(float),
        d["type_factor"].values.astype(float),
        d["target_season_factor"].values.astype(float),
        d["competition_dampener"].values.astype(float),
    ]
    latent_only_terms = [
        1.0 + w * (d["censoring_score"].values * CENSORING_UPLIFT),
        np.where(w > 1e-9, 1.0 + w * (d["peer_efficiency_gap"].values - 1.0), 1.0),
        1.0 + w * (d["effective_catchment_score"].values * CATCHMENT_UPLIFT),
    ]

    running = d["jan_base"].values.astype(float)
    base_values = np.log1p(np.clip(running, 0.0, None))
    all_terms = shared_terms + latent_only_terms
    contribs = np.zeros((n, len(all_terms)), dtype=float)

    for j, term in enumerate(all_terms):
        term = np.clip(term, 1e-9, None)
        next_running = running * term
        contribs[:, j] = np.log1p(np.clip(next_running, 0.0, None)) - np.log1p(np.clip(running, 0.0, None))
        running = next_running

    raw = compute_heuristic_potential(d)
    final_liters = finalize_latent_predictions(raw, d)
    prediction_log = np.log1p(np.clip(final_liters, 0.0, None))

    chain_log = base_values + contribs.sum(axis=1)
    residual = prediction_log - chain_log
    feature_names = list(COMPONENT_NAMES)
    if np.any(np.abs(residual) > 1e-6):
        feature_names = feature_names + ["Calibration (floor/cap/blend)"]
        contribs = np.column_stack([contribs, residual])

    result = {
        "explanation_type": "heuristic_waterfall_two_regime",
        "regime_weight_mean": float(np.mean(w)),
        "shap_values": contribs,
        "base_values": base_values,
        "base_value": float(np.mean(base_values)),
        "feature_names": feature_names,
        "prediction_log": prediction_log,
        "X_pred": d,
    }
    if "Outlet_ID" in d.columns:
        result["Outlet_ID"] = d["Outlet_ID"].values
    return result
