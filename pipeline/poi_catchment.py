"""
DataStorm 2026 - POI catchment feature enrichment
=================================================
Restores the legacy weighted poi_* catchment score (p95 scaling) and blends
it with Gaussian/gravity combined_catchment_score for production heuristics.

Block 1: global-max normalization on combined_catchment_score crushed most
outlets toward zero; poi_catchment_score spreads signal across the network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Weighted POI count columns (from 03_poi_scraper poi_{category} integers)
POI_CATCHMENT_WEIGHTS: dict[str, float] = {
    "poi_school": 2.0,
    "poi_bus_stop": 2.0,
    "poi_hospital": 1.5,
    "poi_tourism": 1.8,
    "poi_market": 0.5,
    "poi_place_worship": 1.2,
    "poi_fuel_station": 1.0,
    "poi_restaurant": 1.3,
    "poi_bank_atm": 1.0,
}


def compute_poi_catchment_raw(df: pd.DataFrame) -> pd.Series:
    """Weighted sum of legacy poi_* integer proximity counts."""
    raw = pd.Series(0.0, index=df.index, dtype=float)
    for col, weight in POI_CATCHMENT_WEIGHTS.items():
        if col in df.columns:
            raw = raw + pd.to_numeric(df[col], errors="coerce").fillna(0.0) * weight
    return raw


def enrich_catchment_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add poi_catchment_score and effective_catchment_score to a feature frame.

    effective_catchment_score = max(poi_catchment_score, combined_catchment_score)
    used by latent_heuristic for latent uncapping.
    """
    out = df.copy()

    out["poi_catchment_raw"] = compute_poi_catchment_raw(out)
    p95 = float(out["poi_catchment_raw"].quantile(0.95))
    if p95 <= 0:
        out["poi_catchment_score"] = 0.0
    else:
        out["poi_catchment_score"] = (out["poi_catchment_raw"] / (p95 + 1e-9)).clip(0.0, 1.0)

    if "combined_catchment_score" not in out.columns:
        out["combined_catchment_score"] = 0.0
    else:
        out["combined_catchment_score"] = (
            pd.to_numeric(out["combined_catchment_score"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
        )

    out["effective_catchment_score"] = np.maximum(
        out["poi_catchment_score"].values.astype(float),
        out["combined_catchment_score"].values.astype(float),
    )

    return out
