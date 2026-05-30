"""
DataStorm 2026 - Reusable Data Quality (DQ) Check Library
==========================================================
All checks are parameterizable functions that:
  1. Accept a DataFrame + configuration.
  2. Return (clean_df, rejected_df) tuples.
  3. Tag every rejected record with a `dq_failure_reason` column.

Additionally, `write_dq_manifest()` writes a JSON audit log after each Silver
run, recording timestamps, per-dataset row counts, rejection totals, and
rejection rates — satisfying the production-grade DQ traceability requirement.

This module is imported by every Silver-layer cleaning script to ensure
consistent application of DQ rules across all datasets.
"""

import json
import pandas as pd
import numpy as np
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union, List, Dict, Tuple


# ---------------------------------------------------------------------------
# 1. DUPLICATE CHECK
# ---------------------------------------------------------------------------

def check_duplicates(
    df: pd.DataFrame,
    primary_key: Union[str, List[str]],
    keep: str = "first",
    dataset_name: str = "dataset",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Detect and quarantine duplicate records.

    Parameters
    ----------
    df            : Input DataFrame.
    primary_key   : Single column name or list of columns forming the PK.
    keep          : Which duplicate occurrence to keep ('first', 'last', False).
    dataset_name  : Label used in the failure reason message.

    Returns
    -------
    clean_df, rejected_df
    """
    if isinstance(primary_key, str):
        primary_key = [primary_key]

    is_dup = df.duplicated(subset=primary_key, keep=keep)
    rejected = df[is_dup].copy()
    rejected["dq_failure_reason"] = (
        f"[{dataset_name}] Duplicate record on key: {primary_key}"
    )
    clean = df[~is_dup].copy()
    return clean, rejected


# ---------------------------------------------------------------------------
# 2. NULL / MANDATORY FIELD CHECK
# ---------------------------------------------------------------------------

def check_nulls(
    df: pd.DataFrame,
    mandatory_fields: List[str],
    dataset_name: str = "dataset",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Flag records where any mandatory field is null or empty string.

    Parameters
    ----------
    df                : Input DataFrame.
    mandatory_fields  : List of column names that must be non-null.
    dataset_name      : Label used in the failure reason message.

    Returns
    -------
    clean_df, rejected_df
    """
    mask = pd.Series(False, index=df.index)
    reasons: Dict[int, List[str]] = {}

    for col in mandatory_fields:
        if col not in df.columns:
            continue
        col_null = df[col].isnull() | (df[col].astype(str).str.strip() == "")
        for idx in df.index[col_null]:
            reasons.setdefault(idx, []).append(col)
        mask |= col_null

    rejected = df[mask].copy()
    rejected["dq_failure_reason"] = rejected.index.map(
        lambda i: f"[{dataset_name}] Null/empty mandatory fields: {reasons.get(i, [])}"
    )
    clean = df[~mask].copy()
    return clean, rejected


# ---------------------------------------------------------------------------
# 3. REFERENTIAL INTEGRITY CHECK
# ---------------------------------------------------------------------------

def check_referential_integrity(
    df: pd.DataFrame,
    fk_column: str,
    reference_set: Union[set, pd.Series],
    dataset_name: str = "dataset",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Validate that all values in `fk_column` exist in `reference_set`.

    Parameters
    ----------
    df             : Input DataFrame.
    fk_column      : Column containing foreign-key values.
    reference_set  : Set/Series of valid reference values.
    dataset_name   : Label used in the failure reason message.

    Returns
    -------
    clean_df, rejected_df
    """
    if not isinstance(reference_set, set):
        reference_set = set(reference_set)

    mask = ~df[fk_column].isin(reference_set)
    rejected = df[mask].copy()
    rejected["dq_failure_reason"] = (
        f"[{dataset_name}] Referential integrity violation on column '{fk_column}'"
    )
    clean = df[~mask].copy()
    return clean, rejected


# ---------------------------------------------------------------------------
# 4. VALUE RANGE CHECK
# ---------------------------------------------------------------------------

def check_value_range(
    df: pd.DataFrame,
    column: str,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    allow_zero: bool = True,
    dataset_name: str = "dataset",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Assert that a numeric column falls within [min_val, max_val].

    Parameters
    ----------
    df           : Input DataFrame.
    column       : Numeric column to check.
    min_val      : Minimum acceptable value (inclusive). None = no lower bound.
    max_val      : Maximum acceptable value (inclusive). None = no upper bound.
    allow_zero   : If False, zero values are also flagged.
    dataset_name : Label used in the failure reason message.

    Returns
    -------
    clean_df, rejected_df
    """
    mask = pd.Series(False, index=df.index)

    if min_val is not None:
        mask |= df[column] < min_val
    if max_val is not None:
        mask |= df[column] > max_val
    if not allow_zero:
        mask |= df[column] == 0

    rejected = df[mask].copy()
    rejected["dq_failure_reason"] = (
        f"[{dataset_name}] Column '{column}' value out of range "
        f"[min={min_val}, max={max_val}, allow_zero={allow_zero}]"
    )
    clean = df[~mask].copy()
    return clean, rejected


# ---------------------------------------------------------------------------
# 5. FORMAT / TYPE CHECK
# ---------------------------------------------------------------------------

def check_format(
    df: pd.DataFrame,
    column: str,
    expected_pattern: Optional[str] = None,
    expected_dtype: Optional[type] = None,
    date_format: Optional[str] = None,
    dataset_name: str = "dataset",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Validate that a column conforms to an expected format/type.

    Parameters
    ----------
    df               : Input DataFrame.
    column           : Column to validate.
    expected_pattern : Optional regex pattern string.
    expected_dtype   : Optional Python type (int, float, str).
    date_format      : Optional strptime format string for date validation.
    dataset_name     : Label used in the failure reason message.

    Returns
    -------
    clean_df, rejected_df
    """
    mask = pd.Series(False, index=df.index)

    if expected_pattern:
        pat = re.compile(expected_pattern)
        mask |= ~df[column].astype(str).apply(lambda v: bool(pat.fullmatch(v)))

    if expected_dtype == int:
        def _is_int(v):
            try:
                return float(v) == int(float(v))
            except Exception:
                return False
        mask |= ~df[column].apply(_is_int)

    if date_format:
        def _is_date(v):
            try:
                pd.to_datetime(v, format=date_format)
                return True
            except Exception:
                return False
        mask |= ~df[column].astype(str).apply(_is_date)

    rejected = df[mask].copy()
    rejected["dq_failure_reason"] = (
        f"[{dataset_name}] Column '{column}' format/type violation "
        f"(pattern={expected_pattern}, dtype={expected_dtype}, date_fmt={date_format})"
    )
    clean = df[~mask].copy()
    return clean, rejected


# ---------------------------------------------------------------------------
# 6. COORDINATE BOUNDS CHECK  (domain-specific for Sri Lanka)
# ---------------------------------------------------------------------------

def check_coordinate_bounds(
    df: pd.DataFrame,
    lat_col: str = "Latitude",
    lon_col: str = "Longitude",
    lat_min: float = 5.9,
    lat_max: float = 9.9,
    lon_min: float = 79.6,
    lon_max: float = 82.0,
    dataset_name: str = "dataset",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Validate that geographic coordinates fall within Sri Lanka's bounding box.

    Parameters
    ----------
    df           : Input DataFrame.
    lat_col      : Latitude column name.
    lon_col      : Longitude column name.
    lat_min/max  : Latitude bounds (default: Sri Lanka).
    lon_min/max  : Longitude bounds (default: Sri Lanka).
    dataset_name : Label used in the failure reason message.

    Returns
    -------
    clean_df, rejected_df
    """
    lat_bad = (df[lat_col] < lat_min) | (df[lat_col] > lat_max)
    lon_bad = (df[lon_col] < lon_min) | (df[lon_col] > lon_max)
    mask = lat_bad | lon_bad

    rejected = df[mask].copy()
    rejected["dq_failure_reason"] = (
        f"[{dataset_name}] Coordinates out of Sri Lanka bounding box "
        f"(lat [{lat_min},{lat_max}], lon [{lon_min},{lon_max}])"
    )
    clean = df[~mask].copy()
    return clean, rejected


# ---------------------------------------------------------------------------
# HELPER - Accumulate and store rejected records
# ---------------------------------------------------------------------------

def accumulate_rejected(
    existing_rejected: pd.DataFrame,
    new_rejected: pd.DataFrame,
) -> pd.DataFrame:
    """Concatenate rejected record batches into a running store."""
    if new_rejected.empty:
        return existing_rejected
    return pd.concat([existing_rejected, new_rejected], ignore_index=True)


def save_rejected(
    rejected_df: pd.DataFrame,
    path: str,
) -> None:
    """Persist quarantined records to CSV with failure reason documented."""
    rejected_df.to_csv(path, index=False)
    print(f"  [QUARANTINE] {len(rejected_df)} records -> {path}")


# ---------------------------------------------------------------------------
# DQ RUN MANIFEST  (audit log for the entire Silver run)
# ---------------------------------------------------------------------------

def write_dq_manifest(
    stats: List[Dict],
    manifest_path: Union[str, Path],
) -> None:
    """
    Write a structured JSON audit log summarising the entire Silver DQ run.

    Parameters
    ----------
    stats         : List of dicts, one per dataset. Each dict must contain:
                    {'dataset': str, 'total_rows': int, 'rejected_rows': int}
    manifest_path : Destination path for the JSON manifest file.

    Output JSON structure
    ---------------------
    {
      "run_timestamp":  "2026-05-16T15:00:00+00:00",
      "pipeline_stage": "Silver",
      "datasets": [
        {
          "dataset":          "transactions",
          "total_rows":       1500000,
          "rejected_rows":    3200,
          "clean_rows":       1496800,
          "rejection_rate_%": 0.21
        },
        ...
      ],
      "total_rejected_all_datasets": 3450,
      "total_rows_all_datasets":     1512000
    }
    """
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    enriched = []
    total_rows     = 0
    total_rejected = 0

    for s in stats:
        tr = s.get("total_rows", 0)
        rr = s.get("rejected_rows", 0)
        rate = round(100.0 * rr / tr, 4) if tr > 0 else 0.0
        enriched.append({
            "dataset":          s["dataset"],
            "total_rows":       tr,
            "rejected_rows":    rr,
            "clean_rows":       tr - rr,
            "rejection_rate_%": rate,
        })
        total_rows     += tr
        total_rejected += rr

    manifest = {
        "run_timestamp":               datetime.now(timezone.utc).isoformat(),
        "pipeline_stage":              "Silver",
        "datasets":                    enriched,
        "total_rows_all_datasets":     total_rows,
        "total_rejected_all_datasets": total_rejected,
        "overall_rejection_rate_%":    round(100.0 * total_rejected / total_rows, 4)
                                       if total_rows > 0 else 0.0,
    }

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"  [DQ MANIFEST] Written -> {manifest_path}")
    print(f"  [DQ MANIFEST] Total rows: {total_rows:,}  |  "
          f"Total rejected: {total_rejected:,}  |  "
          f"Overall rejection rate: {manifest['overall_rejection_rate_%']:.3f}%")
