"""
SILVER LAYER - Data Cleaning & Quality Enforcement
====================================================
Reads Bronze Parquet files, applies reusable DQ checks from dq_checks.py,
quarantines failing records into the `rejected/` store (with documented
failure reasons), and writes sanitized Silver Parquet files.

Data Forensics Log
------------------
The following system anomalies were identified and trapped:

TRANSACTIONS:
  [A1] Negative Volume_Liters  - ERP credit-note ghost entries / system reversals.
  [A2] Zero Volume_Liters      - Null-transaction SFA auto-inserts (no actual delivery).
  [A3] Extreme positive volume outliers (> 99.99th pctile & > 3× IQR fence) - likely
       data-entry errors or batch aggregation artefacts.
  [A4] Duplicate records (same Outlet+Year+Month+Distributor+SKU) - SFA sync glitches.
  [A5] Referential integrity: Outlet_ID not in outlet master.

OUTLET_MASTER:
  [B1] Null Outlet_Size (196 records) - master-data decay; imputed from mode per type.
  [B2] Mixed-case / misspelt Outlet_Type (e.g. 'Grocry', ' Eatery ', 'Bakry') -
       normalised to canonical values.
  [B3] Outlet_Size 'small' -> 'Small' (case normalisation).
  [B4] Negative / impossible Cooler_Count - range check.

OUTLET_COORDINATES:
  [C1] Coordinates outside Sri Lanka's bounding box (240 records) - GPS dropouts
       or erroneous ERP entries; quarantined.

DISTRIBUTOR_SEASONALITY:
  [D1] Null or blank Seasonality_Index - mandatory field check.
  [D2] January 2026 not present - will be forward-filled from January 2023-2025 mode
       during Gold feature engineering.

HOLIDAY_LIST:
  [E1] Invalid date formats - format check.

Usage:
    python pipeline/02_silver_clean.py
"""

import sys
import os
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

import pandas as pd
import numpy as np

import dq_checks as dq

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BRONZE_DIR   = ROOT / "pipeline" / "bronze"
SILVER_DIR   = ROOT / "pipeline" / "silver"
REJECTED_DIR = ROOT / "pipeline" / "rejected"
SILVER_DIR.mkdir(parents=True, exist_ok=True)
REJECTED_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Canonical lookup tables (master-data normalisation)
# ---------------------------------------------------------------------------
OUTLET_TYPE_CANONICAL = {
    "grocry":   "Grocery",
    "grocery":  "Grocery",
    "hotel":    "Hotel",
    "pharmacy": "Pharmacy",
    "kiosk":    "Kiosk",
    "eatery":   "Eatery",
    "bakery":   "Bakery",
    "bakry":    "Bakery",
    "smmt":     "SMMT",
}

OUTLET_SIZE_CANONICAL = {
    "small":       "Small",
    "medium":      "Medium",
    "large":       "Large",
    "extra large": "Extra Large",
}

# ---------------------------------------------------------------------------
# Helper: load bronze parquet
# ---------------------------------------------------------------------------
def load_bronze(name: str) -> pd.DataFrame:
    path = BRONZE_DIR / f"{name}.parquet"
    df = pd.read_parquet(path, engine="pyarrow")
    print(f"  Loaded bronze/{name}.parquet  ({len(df):,} rows)")
    return df


# ===========================================================================
# 1. TRANSACTIONS
# ===========================================================================
def clean_transactions() -> dict:
    print("\n--- Cleaning: transactions ---")
    dataset = "transactions"
    rejected_all = pd.DataFrame()

    df = load_bronze(dataset)
    original_count = len(df)

    # Valid outlet IDs (from master)
    outlet_master = load_bronze("outlet_master")
    valid_outlet_ids = set(outlet_master["Outlet_ID"])

    # [A4] Duplicate check
    df, rej = dq.check_duplicates(
        df,
        primary_key=["Outlet_ID", "Year", "Month", "Distributor_ID", "SKU_ID"],
        keep="first",
        dataset_name=dataset,
    )
    rejected_all = dq.accumulate_rejected(rejected_all, rej)
    print(f"    [A4] Duplicates removed: {len(rej):,}")

    # [A5] Referential integrity - Outlet_ID
    df, rej = dq.check_referential_integrity(
        df, fk_column="Outlet_ID",
        reference_set=valid_outlet_ids,
        dataset_name=dataset,
    )
    rejected_all = dq.accumulate_rejected(rejected_all, rej)
    print(f"    [A5] Referential integrity failures: {len(rej):,}")

    # [A1] Negative Volume_Liters (ERP reversals / credit notes)
    df, rej = dq.check_value_range(
        df, column="Volume_Liters",
        min_val=0.001,  # strictly positive
        dataset_name=dataset,
    )
    rejected_all = dq.accumulate_rejected(rejected_all, rej)
    print(f"    [A1+A2] Negative/zero volume records removed: {len(rej):,}")

    # [A3] Extreme outlier cap: IQR fence
    Q1 = df["Volume_Liters"].quantile(0.25)
    Q3 = df["Volume_Liters"].quantile(0.75)
    IQR = Q3 - Q1
    upper_fence = Q3 + 5 * IQR   # 5× IQR -> very conservative to preserve real peaks
    # Also hard cap at 2000 L/month per SKU per outlet (physical max cooler truck)
    hard_cap = 2000.0
    outlier_mask = (df["Volume_Liters"] > upper_fence) | (df["Volume_Liters"] > hard_cap)
    rej_outlier = df[outlier_mask].copy()
    rej_outlier["dq_failure_reason"] = (
        f"[{dataset}] [A3] Extreme volume outlier (>{min(upper_fence, hard_cap):.1f} L)"
    )
    df = df[~outlier_mask].copy()
    rejected_all = dq.accumulate_rejected(rejected_all, rej_outlier)
    print(f"    [A3] Extreme volume outliers removed: {len(rej_outlier):,}")

    # Null checks on mandatory fields
    df, rej = dq.check_nulls(
        df,
        mandatory_fields=["Outlet_ID", "Year", "Month", "Distributor_ID", "SKU_ID",
                          "Volume_Liters", "Total_Bill_Value"],
        dataset_name=dataset,
    )
    rejected_all = dq.accumulate_rejected(rejected_all, rej)

    # Year/Month range check
    df, rej = dq.check_value_range(df, column="Year", min_val=2020, max_val=2026,
                                   dataset_name=dataset)
    rejected_all = dq.accumulate_rejected(rejected_all, rej)
    df, rej = dq.check_value_range(df, column="Month", min_val=1, max_val=12,
                                   dataset_name=dataset)
    rejected_all = dq.accumulate_rejected(rejected_all, rej)

    print(f"    TOTAL rejected: {len(rejected_all):,} / {original_count:,} "
          f"({100*len(rejected_all)/original_count:.2f}%)")
    print(f"    Clean rows: {len(df):,}")

    # Persist
    df.to_parquet(SILVER_DIR / "transactions.parquet", index=False)
    dq.save_rejected(rejected_all, REJECTED_DIR / "transactions_rejected.csv")
    return {"dataset": dataset, "total_rows": original_count, "rejected_rows": len(rejected_all)}


# ===========================================================================
# 2. OUTLET MASTER
# ===========================================================================
def clean_outlet_master() -> dict:
    print("\n--- Cleaning: outlet_master ---")
    dataset = "outlet_master"
    rejected_all = pd.DataFrame()

    df = load_bronze(dataset)
    original_count = len(df)

    # [B3+B2] Normalise Outlet_Type (strip whitespace + lowercase map)
    df["Outlet_Type"] = (
        df["Outlet_Type"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(OUTLET_TYPE_CANONICAL)
    )

    # [B3] Normalise Outlet_Size
    df["Outlet_Size"] = (
        df["Outlet_Size"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(OUTLET_SIZE_CANONICAL)   # NaN for unknown -> handled below
    )

    # [B1] Impute missing Outlet_Size from mode per Outlet_Type
    size_mode = (
        df.dropna(subset=["Outlet_Size"])
        .groupby("Outlet_Type")["Outlet_Size"]
        .agg(lambda x: x.mode().iloc[0] if len(x) > 0 else "Small")
        .to_dict()
    )
    df["Outlet_Size"] = df.apply(
        lambda row: size_mode.get(row["Outlet_Type"], "Small")
        if pd.isna(row["Outlet_Size"]) else row["Outlet_Size"],
        axis=1,
    )

    # [B4] Cooler_Count range check
    df, rej = dq.check_value_range(
        df, column="Cooler_Count", min_val=0, max_val=50,
        dataset_name=dataset,
    )
    rejected_all = dq.accumulate_rejected(rejected_all, rej)
    print(f"    [B4] Bad Cooler_Count: {len(rej):,}")

    # [B?] Duplicate Outlet_ID
    df, rej = dq.check_duplicates(df, primary_key="Outlet_ID", dataset_name=dataset)
    rejected_all = dq.accumulate_rejected(rejected_all, rej)
    print(f"    Duplicate Outlet_IDs: {len(rej):,}")

    # Null check mandatory
    df, rej = dq.check_nulls(df, mandatory_fields=["Outlet_ID", "Outlet_Type"],
                              dataset_name=dataset)
    rejected_all = dq.accumulate_rejected(rejected_all, rej)

    print(f"    TOTAL rejected: {len(rejected_all):,} / {original_count:,}")
    print(f"    Clean rows: {len(df):,}")

    df.to_parquet(SILVER_DIR / "outlet_master.parquet", index=False)
    dq.save_rejected(rejected_all, REJECTED_DIR / "outlet_master_rejected.csv")
    return {"dataset": dataset, "total_rows": original_count, "rejected_rows": len(rejected_all)}


# ===========================================================================
# 3. OUTLET COORDINATES
# ===========================================================================
def clean_outlet_coordinates() -> dict:
    print("\n--- Cleaning: outlet_coordinates ---")
    dataset = "outlet_coordinates"
    rejected_all = pd.DataFrame()

    df = load_bronze(dataset)
    original_count = len(df)

    # [C1] Coordinates outside Sri Lanka
    df, rej = dq.check_coordinate_bounds(
        df, lat_col="Latitude", lon_col="Longitude", dataset_name=dataset
    )
    rejected_all = dq.accumulate_rejected(rejected_all, rej)
    print(f"    [C1] Out-of-bounds coordinates: {len(rej):,}")

    # Null check
    df, rej = dq.check_nulls(df, mandatory_fields=["Outlet_ID", "Latitude", "Longitude"],
                              dataset_name=dataset)
    rejected_all = dq.accumulate_rejected(rejected_all, rej)

    # Duplicate Outlet_ID
    df, rej = dq.check_duplicates(df, primary_key="Outlet_ID", dataset_name=dataset)
    rejected_all = dq.accumulate_rejected(rejected_all, rej)
    print(f"    Duplicate Outlet_IDs: {len(rej):,}")

    print(f"    TOTAL rejected: {len(rejected_all):,} / {original_count:,}")
    print(f"    Clean rows: {len(df):,}")

    df.to_parquet(SILVER_DIR / "outlet_coordinates.parquet", index=False)
    dq.save_rejected(rejected_all, REJECTED_DIR / "outlet_coordinates_rejected.csv")
    return {"dataset": dataset, "total_rows": original_count, "rejected_rows": len(rejected_all)}


# ===========================================================================
# 4. DISTRIBUTOR SEASONALITY
# ===========================================================================
def clean_distributor_seasonality() -> dict:
    print("\n--- Cleaning: distributor_seasonality ---")
    dataset = "distributor_seasonality"
    rejected_all = pd.DataFrame()

    df = load_bronze("distributor_seasonality")
    original_count = len(df)

    valid_dists = {
        "DIST_W_01","DIST_W_02","DIST_W_03",
        "DIST_C_01","DIST_C_02","DIST_C_03",
        "DIST_NW_01","DIST_NW_02",
        "DIST_S_01","DIST_S_02",
    }

    # Referential integrity
    df, rej = dq.check_referential_integrity(
        df, fk_column="Distributor_ID",
        reference_set=valid_dists,
        dataset_name=dataset,
    )
    rejected_all = dq.accumulate_rejected(rejected_all, rej)

    # Null check
    df, rej = dq.check_nulls(
        df,
        mandatory_fields=["Distributor_ID","Year","Month","Seasonality_Index"],
        dataset_name=dataset,
    )
    rejected_all = dq.accumulate_rejected(rejected_all, rej)
    print(f"    [D1] Null seasonality index: {len(rej):,}")

    # Range checks
    df, rej = dq.check_value_range(df, "Year", min_val=2020, max_val=2030,
                                   dataset_name=dataset)
    rejected_all = dq.accumulate_rejected(rejected_all, rej)
    df, rej = dq.check_value_range(df, "Month", min_val=1, max_val=12,
                                   dataset_name=dataset)
    rejected_all = dq.accumulate_rejected(rejected_all, rej)

    # Normalise Seasonality_Index values
    valid_indices = {"Moderate", "Favorable", "Un-Favorable"}
    bad_idx = ~df["Seasonality_Index"].isin(valid_indices)
    rej_idx = df[bad_idx].copy()
    rej_idx["dq_failure_reason"] = (
        f"[{dataset}] Invalid Seasonality_Index value (not in {valid_indices})"
    )
    df = df[~bad_idx].copy()
    rejected_all = dq.accumulate_rejected(rejected_all, rej_idx)
    print(f"    Invalid Seasonality_Index values: {len(rej_idx):,}")

    print(f"    TOTAL rejected: {len(rejected_all):,} / {original_count:,}")
    print(f"    Clean rows: {len(df):,}")

    df.to_parquet(SILVER_DIR / "distributor_seasonality.parquet", index=False)
    dq.save_rejected(rejected_all, REJECTED_DIR / "distributor_seasonality_rejected.csv")
    return {"dataset": dataset, "total_rows": original_count, "rejected_rows": len(rejected_all)}


# ===========================================================================
# 5. HOLIDAY LIST
# ===========================================================================
def clean_holiday_list() -> dict:
    print("\n--- Cleaning: holiday_list ---")
    dataset = "holiday_list"
    rejected_all = pd.DataFrame()

    df = load_bronze("holiday_list")
    original_count = len(df)

    # Parse dates and flag bad formats
    df["Date_parsed"] = pd.to_datetime(df["Date"], errors="coerce", utc=True)
    bad_date_mask = df["Date_parsed"].isna()
    rej = df[bad_date_mask].copy()
    rej["dq_failure_reason"] = f"[{dataset}] [E1] Unparseable date value"
    rejected_all = dq.accumulate_rejected(rejected_all, rej)
    df = df[~bad_date_mask].copy()
    df["Date"] = df["Date_parsed"].dt.date
    df.drop(columns=["Date_parsed"], inplace=True)
    print(f"    [E1] Bad date format records: {len(rej):,}")

    # Null check
    df, rej = dq.check_nulls(df, mandatory_fields=["Date","Holiday_Name"],
                              dataset_name=dataset)
    rejected_all = dq.accumulate_rejected(rejected_all, rej)

    # Duplicate dates
    df, rej = dq.check_duplicates(df, primary_key="Date", dataset_name=dataset)
    rejected_all = dq.accumulate_rejected(rejected_all, rej)
    print(f"    Duplicate date entries: {len(rej):,}")

    print(f"    TOTAL rejected: {len(rejected_all):,} / {original_count:,}")
    print(f"    Clean rows: {len(df):,}")

    df.to_parquet(SILVER_DIR / "holiday_list.parquet", index=False)
    dq.save_rejected(rejected_all, REJECTED_DIR / "holiday_list_rejected.csv")
    return {"dataset": dataset, "total_rows": original_count, "rejected_rows": len(rejected_all)}


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    print("=" * 60)
    print("SILVER LAYER - Data Cleaning & Quality Enforcement")
    print("=" * 60)
    dq_stats = [
        clean_transactions(),
        clean_outlet_master(),
        clean_outlet_coordinates(),
        clean_distributor_seasonality(),
        clean_holiday_list(),
    ]
    # Write DQ run manifest (JSON audit log)
    dq.write_dq_manifest(dq_stats, REJECTED_DIR / "dq_manifest.json")
    print("\n[OK]  Silver cleaning complete.\n")


if __name__ == "__main__":
    main()
