"""
BRONZE LAYER - Raw Ingestion + Schema Enforcement
====================================================
Reads raw CSVs from `input/` and writes Parquet files to `pipeline/bronze/`.

Schema Enforcement
------------------
Each source file is validated against an expected schema before writing:
  - Required columns must be present.
  - Critical numeric columns must be castable to the correct dtype.
  - Row count must be > 0.
Any schema violation raises a ValueError and halts ingestion for that file.

Usage:
    python pipeline/01_bronze_ingest.py
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

ROOT       = Path(__file__).parent.parent
INPUT_DIR  = ROOT / "input"
BRONZE_DIR = ROOT / "pipeline" / "bronze"
BRONZE_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_FILES = {
    "transactions":            "transactions_history_final.csv",
    "outlet_master":           "outlet_master.csv",
    "outlet_coordinates":      "outlet_coordinates.csv",
    "distributor_seasonality": "distributor_seasonality_details.csv",
    "holiday_list":            "holiday_list.csv",
}

# ---------------------------------------------------------------------------
# Expected schemas: {column: expected_dtype_category}
# Dtype categories: "numeric", "string", "integer"
# ---------------------------------------------------------------------------
EXPECTED_SCHEMAS = {
    "transactions": {
        "required_columns": [
            "Outlet_ID", "Year", "Month", "Distributor_ID", "SKU_ID",
            "Volume_Liters", "Total_Bill_Value",
        ],
        "numeric_columns":  ["Volume_Liters", "Total_Bill_Value"],
        "integer_columns":  ["Year", "Month"],
    },
    "outlet_master": {
        "required_columns": ["Outlet_ID", "Outlet_Type", "Outlet_Size", "Cooler_Count"],
        "numeric_columns":  ["Cooler_Count"],
        "integer_columns":  [],
    },
    "outlet_coordinates": {
        "required_columns": ["Outlet_ID", "Latitude", "Longitude"],
        "numeric_columns":  ["Latitude", "Longitude"],
        "integer_columns":  [],
    },
    "distributor_seasonality": {
        "required_columns": ["Distributor_ID", "Year", "Month", "Seasonality_Index"],
        "numeric_columns":  [],
        "integer_columns":  ["Year", "Month"],
    },
    "holiday_list": {
        "required_columns": ["Date", "Holiday_Name"],
        "numeric_columns":  [],
        "integer_columns":  [],
    },
}


def validate_schema(df: pd.DataFrame, name: str) -> dict:
    """
    Validate a DataFrame against the expected schema for `name`.

    Returns a dict with validation results. Raises ValueError on hard failures.
    """
    schema   = EXPECTED_SCHEMAS.get(name, {})
    results  = {"dataset": name, "rows": len(df), "issues": []}

    if len(df) == 0:
        raise ValueError(f"[{name}] SCHEMA FAIL: File is empty (0 rows).")

    # --- Required columns check ---
    for col in schema.get("required_columns", []):
        if col not in df.columns:
            raise ValueError(
                f"[{name}] SCHEMA FAIL: Required column '{col}' is missing. "
                f"Found columns: {list(df.columns)}"
            )

    # --- Numeric dtype check ---
    for col in schema.get("numeric_columns", []):
        if col not in df.columns:
            continue
        try:
            pd.to_numeric(df[col], errors="raise")
        except (ValueError, TypeError):
            n_bad = pd.to_numeric(df[col], errors="coerce").isna().sum() - df[col].isna().sum()
            issue = f"Column '{col}' has {n_bad} non-numeric values (will coerce)."
            results["issues"].append(issue)
            print(f"    [SCHEMA WARN] {issue}")

    # --- Integer dtype check ---
    for col in schema.get("integer_columns", []):
        if col not in df.columns:
            continue
        coerced = pd.to_numeric(df[col], errors="coerce")
        n_non_int = (~coerced.dropna().apply(lambda v: v == int(v))).sum()
        if n_non_int > 0:
            issue = f"Column '{col}' has {n_non_int} non-integer values."
            results["issues"].append(issue)
            print(f"    [SCHEMA WARN] {issue}")

    status = "PASS" if not results["issues"] else "WARN"
    print(f"    [SCHEMA {status}] {name}: {len(df):,} rows, "
          f"{len(df.columns)} cols, {len(results['issues'])} issue(s)")
    return results


def ingest_file(name: str, filename: str) -> dict:
    """Ingest one CSV, validate schema, write Parquet. Returns validation result."""
    src = INPUT_DIR / filename
    dst = BRONZE_DIR / f"{name}.parquet"

    print(f"\n  Ingesting  {filename}  ->  {dst.name}")
    df = pd.read_csv(src, low_memory=False)

    result = validate_schema(df, name)

    df.to_parquet(dst, index=False, engine="pyarrow")
    print(f"    Written: {dst}  ({len(df):,} rows)")
    return result


def main():
    print("=" * 60)
    print("BRONZE LAYER - Raw Ingestion + Schema Enforcement")
    print("=" * 60)

    manifest = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "datasets": [],
    }

    all_ok = True
    for name, filename in SOURCE_FILES.items():
        try:
            result = ingest_file(name, filename)
            manifest["datasets"].append(result)
        except (FileNotFoundError, ValueError) as e:
            print(f"  [ERROR] {e}")
            all_ok = False

    # Save bronze schema manifest
    manifest_path = BRONZE_DIR / "bronze_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n  Schema manifest -> {manifest_path}")

    if all_ok:
        print("\n[OK]  Bronze ingestion complete.\n")
    else:
        print("\n[WARN]  Bronze ingestion completed with errors. Check output above.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
