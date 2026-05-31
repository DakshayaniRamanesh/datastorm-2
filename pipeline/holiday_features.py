"""
DataStorm 2026 - Holiday calendar enrichment
============================================
Classify Sri Lanka public holidays into weighted tiers for demand features:
  - high_effect: major footfall / stocking events (New Year, Vesak, Christmas, …)
  - poya: monthly full-moon Poya days (baseline religious calendar rhythm)
  - standard: other public / mercantile holidays

Source rows often repeat the same calendar day per Holiday_Type (Public, Bank, …);
silver dedupes to one logical event per date before feature aggregation.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

# Tier weights (used in monthly weighted demand signal)
POYA_WEIGHT = 1.0
HIGH_EFFECT_WEIGHT = 2.0
STANDARD_WEIGHT = 0.5

HIGH_EFFECT_KEYWORDS = (
    "new year",
    "sinhala and tamil",
    "tamil thai pongal",
    "pongal",
    "national day",
    "independence",
    "vesak",
    "good friday",
    "labour day",
    "labor day",
    "christmas",
    "deepavali",
    "diwali",
    "milad",
    "prophet",
    "hadji",
    "hajj",
    "ramadan",
    "eid",
    "festival",
)

ISO_DATE_PREFIX = re.compile(r"^(\d{4}-\d{2}-\d{2})")


def parse_holiday_dates(series: pd.Series) -> pd.Series:
    """Parse competition ISO8601 / plain dates without dayfirst mangling."""
    raw = series.astype(str).str.strip()
    # Strip trailing Z and time portion for robust parsing
    cleaned = raw.str.replace("Z", "", regex=False)
    cleaned = cleaned.str.replace(r"T\d{2}:\d{2}:\d{2}", "", regex=True)
    prefix = cleaned.str.extract(ISO_DATE_PREFIX, expand=False)
    parsed = pd.to_datetime(prefix, errors="coerce")
    missing = parsed.isna()
    if missing.any():
        parsed = parsed.fillna(pd.to_datetime(cleaned[missing], errors="coerce", utc=True).dt.tz_localize(None))
    return parsed


def classify_holiday(name: str) -> Tuple[str, float]:
    """Return (tier, weight) for a holiday name."""
    n = (name or "").lower()
    if any(k in n for k in HIGH_EFFECT_KEYWORDS):
        return "high_effect", HIGH_EFFECT_WEIGHT
    if "poya" in n:
        return "poya", POYA_WEIGHT
    return "standard", STANDARD_WEIGHT


def enrich_holiday_tiers(df: pd.DataFrame) -> pd.DataFrame:
    """Add holiday_tier and holiday_weight from Holiday_Name."""
    out = df.copy()
    tiers_weights = out["Holiday_Name"].astype(str).map(classify_holiday)
    out["holiday_tier"] = [t for t, _ in tiers_weights]
    out["holiday_weight"] = [w for _, w in tiers_weights]
    return out


def dedupe_calendar_days(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per calendar date — source repeats same event for Public/Bank/Mercantile.
    Keeps the row with highest weight; merges tier flags when needed.
    """
    if df.empty:
        return df

    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"]).dt.normalize()

    agg = (
        out.sort_values("holiday_weight", ascending=False)
        .groupby("Date", as_index=False)
        .agg(
            Holiday_Name=("Holiday_Name", "first"),
            Holiday_Type=("Holiday_Type", "first"),
            holiday_tier=("holiday_tier", "first"),
            holiday_weight=("holiday_weight", "max"),
            is_poya=("holiday_tier", lambda s: int((s == "poya").any())),
            is_high_effect=("holiday_tier", lambda s: int((s == "high_effect").any())),
        )
    )
    # High-effect Poya (e.g. Vesak) should count in both buckets for features
    poya_mask = agg["Holiday_Name"].str.lower().str.contains("poya", na=False)
    agg.loc[poya_mask, "is_poya"] = 1
    hi_mask = agg["Holiday_Name"].astype(str).map(lambda n: classify_holiday(n)[0] == "high_effect")
    agg.loc[hi_mask, "is_high_effect"] = 1
    return agg


def prepare_silver_holidays(df: pd.DataFrame) -> pd.DataFrame:
    """Parse, classify, and dedupe raw bronze holiday rows for silver output."""
    out = df.copy()
    out["Date"] = parse_holiday_dates(out["Date"])
    out = out[out["Date"].notna()].copy()
    out = enrich_holiday_tiers(out)
    out = dedupe_calendar_days(out)
    out["Date"] = out["Date"].dt.date
    return out.reset_index(drop=True)


def month_holiday_features(
    holidays: pd.DataFrame,
    year: int,
    month: int,
) -> Dict[str, Any]:
    """Aggregate weighted holiday signals for a single calendar month."""
    stats = _month_holiday_features_core(holidays, year, month)

    # Competition calendar often ends before forecast year — proxy Jan 2026 from prior Januaries
    if stats["holiday_total_count"] == 0 and year >= 2026 and month == 1 and holidays is not None:
        prior = [
            _month_holiday_features_core(holidays, y, 1)
            for y in range(2023, year)
            if _month_holiday_features_core(holidays, y, 1)["holiday_total_count"] > 0
        ]
        if prior:
            stats = {
                "holiday_total_count": int(round(np.mean([p["holiday_total_count"] for p in prior]))),
                "holiday_poya_count": int(round(np.mean([p["holiday_poya_count"] for p in prior]))),
                "holiday_high_effect_count": int(round(np.mean([p["holiday_high_effect_count"] for p in prior]))),
                "holiday_weighted_score": round(float(np.mean([p["holiday_weighted_score"] for p in prior])), 2),
                "high_holiday_month": int(np.mean([p["high_holiday_month"] for p in prior]) >= 0.5),
                "jan_holiday_count": int(round(np.mean([p["jan_holiday_count"] for p in prior]))),
            }

    return stats


def _month_holiday_features_core(
    holidays: pd.DataFrame,
    year: int,
    month: int,
) -> Dict[str, Any]:
    """Aggregate weighted holiday signals for a single calendar month."""
    if holidays is None or holidays.empty:
        return {
            "holiday_total_count": 0,
            "holiday_poya_count": 0,
            "holiday_high_effect_count": 0,
            "holiday_weighted_score": 0.0,
            "high_holiday_month": 0,
            # legacy aliases
            "jan_holiday_count": 0,
        }

    h = holidays.copy()
    h["Date"] = pd.to_datetime(h["Date"])
    mask = (h["Date"].dt.year == year) & (h["Date"].dt.month == month)
    month_df = h.loc[mask]

    if month_df.empty:
        return {
            "holiday_total_count": 0,
            "holiday_poya_count": 0,
            "holiday_high_effect_count": 0,
            "holiday_weighted_score": 0.0,
            "high_holiday_month": 0,
            "jan_holiday_count": 0,
        }

    # Unique calendar days in month
    month_df = month_df.drop_duplicates(subset=["Date"])

    poya_count = int(month_df.get("is_poya", (month_df["holiday_tier"] == "poya").astype(int)).sum())
    hi_count = int(month_df.get("is_high_effect", (month_df["holiday_tier"] == "high_effect").astype(int)).sum())
    total = len(month_df)
    weighted = float(month_df["holiday_weight"].sum()) if "holiday_weight" in month_df.columns else float(total)

    return {
        "holiday_total_count": total,
        "holiday_poya_count": poya_count,
        "holiday_high_effect_count": hi_count,
        "holiday_weighted_score": round(weighted, 2),
        "high_holiday_month": int(weighted >= 4.0 or hi_count >= 2),
        "jan_holiday_count": total,
    }


def load_silver_holidays(path) -> Optional[pd.DataFrame]:
    """Load prepared silver holiday calendar if present."""
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def apply_holiday_features_to_frame(
    feats: pd.DataFrame,
    holidays: Optional[pd.DataFrame],
    year: int,
    month: int,
    *,
    jan_prefix: bool = False,
) -> pd.DataFrame:
    """Attach holiday columns to an outlet feature frame (broadcast same month values)."""
    stats = month_holiday_features(holidays, year, month)
    out = feats.copy()
    for key, val in stats.items():
        col = f"jan_{key}" if jan_prefix and not key.startswith("jan_") else key
        if jan_prefix and key == "jan_holiday_count":
            col = "jan_holiday_count"
        out[col] = val

    if jan_prefix:
        out["jan_holiday_poya_count"] = stats["holiday_poya_count"]
        out["jan_holiday_high_effect_count"] = stats["holiday_high_effect_count"]
        out["jan_holiday_weighted_score"] = stats["holiday_weighted_score"]
    return out
