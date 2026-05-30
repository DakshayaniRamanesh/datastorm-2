"""
GOLD LAYER - Feature Engineering + LightGBM Demand Forecasting
===============================================================
Reads all Silver-layer datasets (plus POI features) and produces:
  1. A model-ready Gold feature table (per outlet).
  2. Final Maximum_Monthly_Liters predictions for January 2026.
  3. A censoring analysis decile breakdown CSV for interpretability.

Methodology: Hybrid LightGBM Latent Demand Estimation
======================================================

PROBLEM FRAMING
---------------
Observed monthly volume Y_obs is left-censored:
    Y_obs = min(D_true, C)
where D_true is true latent demand and C is the binding systemic constraint
(credit limit, delivery cap, or stockout).

HYBRID APPROACH
---------------
Phase 1 — Feature Engineering (preserved exactly):
  All domain signals are computed as before:
  - 5-signal composite censoring score (plateau, dist-cap, stagnation, CV, Q4 suppression)
  - 3 behavioral features (capacity proximity, purchase pace variance, dist rank)
  - POI catchment score, peer efficiency gap (SFA proxy)
  - Outlet size/type indicators, seasonality, historical volume statistics

Phase 2 — LightGBM Learnable Demand Function (replaces multiplicative formula):
  A LightGBM regressor is trained on a panel of historical monthly records.
  Target = log1p(observed monthly volume) for each (outlet, month) pair.
  The model learns nonlinear relationships and feature interactions that
  the handcrafted multiplier formula cannot capture.

Phase 3 — Prediction & Light Calibration:
  Final Maximum_Monthly_Liters = expm1(LGB_prediction)
  Optionally adjusted by a censoring-aware scaling factor for interpretability.

Usage:
    python pipeline/04_gold_features_model.py
"""

import sys
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import lightgbm as lgb
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

ROOT      = Path(__file__).parent.parent
SILVER    = ROOT / "pipeline" / "silver"
POI_CACHE = ROOT / "pipeline" / "poi_cache"
GOLD_DIR  = ROOT / "pipeline" / "gold"
OUTPUT    = ROOT / "output"
GOLD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEASONALITY_MULTIPLIER = {
    "Favorable":    1.15,
    "Moderate":     1.00,
    "Un-Favorable": 0.88,
}

SIZE_POTENTIAL_FACTOR = {
    "Extra Large": 1.30,
    "Large":       1.15,
    "Medium":      1.00,
    "Small":       0.88,
}

TYPE_POTENTIAL_FACTOR = {
    "Grocery":  1.10,
    "Hotel":    1.20,
    "Pharmacy": 0.90,
    "Kiosk":    0.85,
    "Eatery":   1.05,
    "Bakery":   0.95,
    "SMMT":     1.25,
}

# LightGBM hyperparameters (tuned for this dataset size)
LGB_PARAMS = {
    "objective":        "regression",
    "metric":           "rmse",
    "n_estimators":     600,
    "learning_rate":    0.03,
    "num_leaves":       63,
    "max_depth":        -1,
    "min_child_samples":20,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":        0.1,
    "reg_lambda":       0.2,
    "verbose":          -1,
    "n_jobs":           -1,
    "random_state":     42,
}

# Number of training months to use as target observations
# We train on the last N months (excluding Jan 2026) as (features → target) pairs
N_TRAINING_MONTHS = 12   # last 12 months of history used as training targets

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def peer_efficiency_gap(outlet_vol: float, peer_vols: np.ndarray,
                         frontier_pctile: int = 90) -> float:
    """
    Stochastic Frontier Analysis (SFA) proxy.
    Returns the ratio of the peer-frontier volume to the outlet's own volume.
    Values > 1.0 indicate the outlet is underperforming versus its peer group.
    """
    if len(peer_vols) < 3:
        return 1.0
    frontier = np.percentile(peer_vols, frontier_pctile)
    if outlet_vol <= 0:
        return 1.5
    ratio = frontier / outlet_vol
    return float(np.clip(ratio, 1.0, 3.0))   # hard cap at 3x


# ---------------------------------------------------------------------------
# VECTORIZED FEATURE BUILDER
# ---------------------------------------------------------------------------

def build_outlet_features(monthly: pd.DataFrame, target_month: int = 1) -> pd.DataFrame:
    """
    Compute per-outlet historical features using vectorized pandas operations.
    This replaces the slow groupby.apply() loop with O(N) vectorised ops.

    Parameters
    ----------
    monthly      : Monthly aggregated transactions with dist_month_median and
                   dist_month_rank already computed.
    target_month : The month number we are predicting for (used for jan_hist_mean).

    Returns
    -------
    DataFrame indexed by Outlet_ID with all engineered signals.
    """
    df = monthly.copy().sort_values(["Outlet_ID", "Year", "Month"])

    gp = df.groupby("Outlet_ID")

    # --- Basic stats ---
    hist_mean_vol   = gp["monthly_volume"].mean()
    hist_median_vol = gp["monthly_volume"].median()
    hist_max_vol    = gp["monthly_volume"].max()
    hist_p75_vol    = gp["monthly_volume"].quantile(0.75)
    hist_p90_vol    = gp["monthly_volume"].quantile(0.90)
    hist_std_vol    = gp["monthly_volume"].std(ddof=0).fillna(0)
    hist_months     = gp["monthly_volume"].count()
    hist_cv         = hist_std_vol / (hist_mean_vol + 1e-9)

    # --- SIGNAL 1: Plateau / rolling 3-month CV < 10% ---
    # Use population std (ddof=0) via algebraic identity: Var = E[X^2] - E[X]^2
    df["monthly_volume_sq"] = df["monthly_volume"].pow(2)
    
    roll3_mean = (
        df.groupby("Outlet_ID")["monthly_volume"]
        .rolling(3, min_periods=3)
        .mean()
        .reset_index(level=0, drop=True)
    )
    roll3_mean_sq = (
        df.groupby("Outlet_ID")["monthly_volume_sq"]
        .rolling(3, min_periods=3)
        .mean()
        .reset_index(level=0, drop=True)
    )
    roll3_std     = np.sqrt((roll3_mean_sq - roll3_mean.pow(2)).clip(lower=0))
    roll3_cv      = roll3_std / (roll3_mean + 1e-9)

    df["plateau_flag"] = (roll3_cv < 0.10).astype(float)
    # Outlets with < 3 months can't have a rolling window — set to NaN (ignored in mean)
    df.loc[df["Outlet_ID"].isin(hist_months[hist_months < 3].index), "plateau_flag"] = np.nan
    cens_plateau  = df.groupby("Outlet_ID")["plateau_flag"].mean().fillna(0.0)

    # --- SIGNAL 2: Tracking distributor delivery cap ---
    df["at_cap_flag"] = (
        np.abs(df["monthly_volume"] - df["dist_month_median"])
        / (df["dist_month_median"] + 1e-9) < 0.15
    ).astype(float)
    cens_dist_cap = df.groupby("Outlet_ID")["at_cap_flag"].mean().fillna(0.0)

    # --- SIGNAL 3: Inter-year stagnation ---
    yearly = (
        df.groupby(["Outlet_ID", "Year"])["monthly_volume"]
        .mean().reset_index().sort_values(["Outlet_ID", "Year"])
    )
    yearly["growth"] = yearly.groupby("Outlet_ID")["monthly_volume"].pct_change()
    yearly["stagnation_flag"] = np.where(
        yearly["growth"].isna(), np.nan,
        (np.abs(yearly["growth"]) < 0.05).astype(float)
    )
    cens_stagnation = yearly.groupby("Outlet_ID")["stagnation_flag"].mean().fillna(0.0)
    yoy_growth      = yearly.groupby("Outlet_ID")["growth"].mean().fillna(0.0).clip(-0.30, 0.50)

    # --- SIGNAL 4: Low overall CV ---
    cens_cv_score = ((0.30 - hist_cv) / 0.30).clip(lower=0.0)

    # --- SIGNAL 5: Q4 suppression ---
    df["q4_vol"] = np.where(df["Month"].isin([10, 11, 12]), df["monthly_volume"], np.nan)
    q4_mean  = df.groupby("Outlet_ID")["q4_vol"].mean()
    q4_count = df.groupby("Outlet_ID")["q4_vol"].count()
    q4_prem  = q4_mean / (hist_mean_vol + 1e-9) - 1.0
    cens_q4_raw = (-q4_prem * 2.0).clip(0.0, 1.0).fillna(0.0)
    cens_q4_suppression = pd.Series(
        np.where(q4_count >= 2, cens_q4_raw, 0.0),
        index=hist_mean_vol.index
    )

    # --- COMPOSITE CENSORING SCORE ---
    censoring_score = (
        0.30 * cens_plateau
        + 0.25 * cens_dist_cap
        + 0.20 * cens_stagnation
        + 0.15 * cens_cv_score
        + 0.10 * cens_q4_suppression
    ).clip(0.0, 1.0)

    # --- FEATURE 6: Capacity Proximity Ratio ---
    roll3_mean_avg          = roll3_mean.groupby(df["Outlet_ID"]).mean()
    capacity_proximity_ratio = np.where(
        hist_months >= 3,
        roll3_mean_avg / (hist_max_vol + 1e-9),
        hist_mean_vol / (hist_max_vol + 1e-9)
    )
    capacity_proximity_ratio = pd.Series(
        np.clip(capacity_proximity_ratio, 0.0, 1.0),
        index=hist_mean_vol.index
    )

    # --- FEATURE 7: Purchase Pace Variance (rolling 6-month, population std) ---
    roll6_mean = (
        df.groupby("Outlet_ID")["monthly_volume"]
        .rolling(6, min_periods=6)
        .mean()
        .reset_index(level=0, drop=True)
    )
    roll6_mean_sq = (
        df.groupby("Outlet_ID")["monthly_volume_sq"]
        .rolling(6, min_periods=6)
        .mean()
        .reset_index(level=0, drop=True)
    )
    roll6_std     = np.sqrt((roll6_mean_sq - roll6_mean.pow(2)).clip(lower=0))
    roll6_cv      = roll6_std / (roll6_mean + 1e-9)
    ppv_raw       = roll6_cv.groupby(df["Outlet_ID"]).mean()
    purchase_pace_variance = pd.Series(
        np.where(hist_months >= 6, ppv_raw, hist_cv),
        index=hist_mean_vol.index
    )

    # --- FEATURE 8: Distributor Relative Rank ---
    dist_rank_mean = df.groupby("Outlet_ID")["dist_month_rank"].mean()

    # --- Target-month historical mean ---
    target_vols = df[df["Month"] == target_month]
    target_mean = (
        target_vols.groupby("Outlet_ID")["monthly_volume"].mean()
        .reindex(hist_mean_vol.index).fillna(hist_mean_vol)
    )

    # --- Primary distributor ---
    dist_sums    = (
        df.groupby(["Outlet_ID", "Distributor_ID"])["monthly_volume"]
        .sum().reset_index()
    )
    primary_dist = (
        dist_sums.sort_values("monthly_volume")
        .groupby("Outlet_ID")["Distributor_ID"].last()
    )

    out = pd.DataFrame({
        "hist_mean_vol":            hist_mean_vol,
        "hist_median_vol":          hist_median_vol,
        "hist_max_vol":             hist_max_vol,
        "hist_p75_vol":             hist_p75_vol,
        "hist_p90_vol":             hist_p90_vol,
        "hist_std_vol":             hist_std_vol,
        "hist_cv":                  hist_cv,
        "hist_months":              hist_months,
        "censoring_score":          censoring_score,
        "cens_plateau":             cens_plateau,
        "cens_dist_cap":            cens_dist_cap,
        "cens_stagnation":          cens_stagnation,
        "cens_cv_score":            cens_cv_score,
        "cens_q4_suppression":      cens_q4_suppression,
        "yoy_growth":               yoy_growth,
        "jan_hist_mean":            target_mean,
        "capacity_proximity_ratio": capacity_proximity_ratio,
        "purchase_pace_variance":   purchase_pace_variance,
        "dist_rank_mean":           dist_rank_mean,
        "primary_dist":             primary_dist,
    }).reset_index()

    return out


# ---------------------------------------------------------------------------
# TRAINING DATASET BUILDER
# ---------------------------------------------------------------------------

def build_training_records(
    monthly: pd.DataFrame,
    outlet: pd.DataFrame,
    season: pd.DataFrame,
    poi_df: pd.DataFrame,
    poi_cols: list,
    n_months: int = N_TRAINING_MONTHS,
) -> pd.DataFrame:
    """
    Build a panel training dataset of (outlet, target_month) records.

    For each of the last `n_months` months in the history:
      - Build features using all data STRICTLY BEFORE that month (no leakage).
      - Record the observed volume in that month as the target (log1p).

    Returns a DataFrame with columns: [feature_cols..., target_log_vol].
    """
    # Determine the training target months (latest N months in data)
    all_periods = sorted(
        monthly[["Year", "Month"]].drop_duplicates()
        .apply(lambda r: (int(r["Year"]), int(r["Month"])), axis=1)
        .tolist()
    )
    # Exclude Jan 2026 if present (prediction target)
    all_periods = [p for p in all_periods if p != (2026, 1)]
    target_periods = all_periods[-n_months:]

    print(f"    Training target periods: {target_periods[0]} -> {target_periods[-1]}")

    records = []
    for (yr, mo) in target_periods:
        # Training cutoff: everything strictly before this (yr, mo)
        cutoff_period = yr * 12 + mo
        train_monthly = monthly[
            monthly["Year"] * 12 + monthly["Month"] < cutoff_period
        ].copy()

        if train_monthly.empty or train_monthly["Outlet_ID"].nunique() < 10:
            continue

        # Drop pre-existing dist stat columns to avoid duplicate cols after merge
        train_monthly = train_monthly.drop(
            columns=[c for c in ["dist_month_median", "dist_month_rank"]
                     if c in train_monthly.columns]
        )

        # Recompute dist stats on this training window only (no leakage)
        dist_med_train = (
            train_monthly.groupby(["Distributor_ID", "Month"])["monthly_volume"]
            .median().rename("dist_month_median").reset_index()
        )
        train_monthly = train_monthly.merge(
            dist_med_train, on=["Distributor_ID", "Month"], how="left"
        )
        train_monthly["dist_month_rank"] = (
            train_monthly.groupby(["Distributor_ID", "Month"])["monthly_volume"]
            .rank(pct=True)
        )

        # Build features
        feats = build_outlet_features(train_monthly, target_month=mo)

        # Merge outlet meta
        feats = feats.merge(
            outlet[["Outlet_ID", "Outlet_Type", "Outlet_Size", "Cooler_Count"]],
            on="Outlet_ID", how="left"
        )

        # Merge seasonality for this target month
        mo_season = (
            season[season["Month"] == mo]
            .groupby("Distributor_ID")["Seasonality_Index"]
            .agg(lambda x: x.mode().iloc[0] if len(x) > 0 else "Moderate")
            .reset_index()
            .rename(columns={"Seasonality_Index": "target_seasonality"})
        )
        feats = feats.merge(
            mo_season, left_on="primary_dist", right_on="Distributor_ID",
            how="left", suffixes=("", "_s")
        )
        feats["target_seasonality"]  = feats["target_seasonality"].fillna("Moderate")
        feats["target_season_factor"] = feats["target_seasonality"].map(
            SEASONALITY_MULTIPLIER
        ).fillna(1.0)
        feats["target_month"] = mo

        # Merge POI if available
        if poi_cols:
            feats = feats.merge(
                poi_df[["Outlet_ID"] + poi_cols], on="Outlet_ID", how="left"
            )
            for c in poi_cols:
                feats[c] = feats[c].fillna(0)

        # Observed volume in this month (target)
        actual = (
            monthly[(monthly["Year"] == yr) & (monthly["Month"] == mo)]
            .groupby("Outlet_ID")["monthly_volume"].sum()
            .reset_index()
            .rename(columns={"monthly_volume": "actual_vol"})
        )
        feats = feats.merge(actual, on="Outlet_ID", how="inner")
        feats = feats[feats["actual_vol"] > 0].copy()
        feats["target_log_vol"] = np.log1p(feats["actual_vol"])

        records.append(feats)

    if not records:
        raise ValueError("No training records could be built. Check data coverage.")

    train_df = pd.concat(records, ignore_index=True)
    print(f"    Training records built: {len(train_df):,} "
          f"({train_df['Outlet_ID'].nunique():,} unique outlets)")
    return train_df


# ---------------------------------------------------------------------------
# FEATURE COLUMN SELECTOR
# ---------------------------------------------------------------------------

def get_feature_cols(df: pd.DataFrame, poi_cols: list) -> list:
    """Return the ordered list of feature columns to use for LightGBM."""
    base_features = [
        # Historical volume statistics
        "hist_mean_vol", "hist_median_vol", "hist_max_vol",
        "hist_p75_vol", "hist_p90_vol", "hist_std_vol",
        "hist_cv", "hist_months",
        # Censoring signals
        "censoring_score", "cens_plateau", "cens_dist_cap",
        "cens_stagnation", "cens_cv_score", "cens_q4_suppression",
        # Behavioral signals
        "capacity_proximity_ratio", "purchase_pace_variance", "dist_rank_mean",
        # Trend
        "yoy_growth",
        # Contextual
        "target_season_factor", "target_month",
        # Outlet characteristics
        "Cooler_Count",
    ]
    # Add POI features if available
    poi_feats = [c for c in poi_cols if c in df.columns]
    base_features += poi_feats
    return [c for c in base_features if c in df.columns]


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("GOLD LAYER - Feature Engineering & LightGBM Demand Model")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # 1. Load Silver datasets
    # -----------------------------------------------------------------------
    print("\n[1/9] Loading Silver datasets...")
    tx       = pd.read_parquet(SILVER / "transactions.parquet")
    outlet   = pd.read_parquet(SILVER / "outlet_master.parquet")
    coords   = pd.read_parquet(SILVER / "outlet_coordinates.parquet")
    season   = pd.read_parquet(SILVER / "distributor_seasonality.parquet")
    holidays = pd.read_parquet(SILVER / "holiday_list.parquet")

    print(f"  Transactions: {len(tx):,}")
    print(f"  Outlets:      {len(outlet):,}")
    print(f"  Coordinates:  {len(coords):,}")

    # -----------------------------------------------------------------------
    # 2. Aggregate transactions to monthly outlet level
    # -----------------------------------------------------------------------
    print("\n[2/9] Aggregating transactions to monthly outlet level...")
    monthly = (
        tx.groupby(["Outlet_ID", "Year", "Month", "Distributor_ID"])
        .agg(
            monthly_volume=(  "Volume_Liters",   "sum"),
            monthly_revenue=( "Total_Bill_Value", "sum"),
            sku_count=(       "SKU_ID",           "nunique"),
            transaction_count=("SKU_ID",          "count"),
        )
        .reset_index()
    )
    print(f"  Monthly records: {len(monthly):,}")

    # -----------------------------------------------------------------------
    # 3. Distributor-month statistics
    # -----------------------------------------------------------------------
    print("\n[3/9] Computing distributor-level delivery benchmarks...")
    dist_month_median = (
        monthly.groupby(["Distributor_ID", "Month"])["monthly_volume"]
        .median()
        .rename("dist_month_median")
        .reset_index()
    )
    monthly = monthly.merge(dist_month_median, on=["Distributor_ID", "Month"], how="left")
    monthly["dist_month_rank"] = (
        monthly.groupby(["Distributor_ID", "Month"])["monthly_volume"]
        .rank(pct=True)
    )

    # -----------------------------------------------------------------------
    # 4. Load POI features (if available, else zero-fill)
    # -----------------------------------------------------------------------
    print("\n[4/9] Loading POI features...")
    poi_path = POI_CACHE / "poi_features.parquet"
    if poi_path.exists():
        poi = pd.read_parquet(poi_path)
        poi_cols = [c for c in poi.columns
                    if c.startswith("poi_") and c not in ["poi_lat", "poi_lon"]]
        h3_col = ["h3_index"] if "h3_index" in poi.columns else []
        print(f"  POI feature columns: {poi_cols}")
        print(f"  Outlets with POI data: {len(poi):,}")
    else:
        print("  [INFO] POI features not found - run 03_poi_scraper.py to enrich.")
        poi      = coords.copy()
        poi_cols = []
        h3_col   = []

    # -----------------------------------------------------------------------
    # 5. Build Gold feature table (full history → Jan 2026 features)
    # -----------------------------------------------------------------------
    print("\n[5/9] Building Gold feature table (full history)...")
    outlet_hist = build_outlet_features(monthly, target_month=1)

    n_constrained = (outlet_hist["censoring_score"] > 0.30).sum()
    print(f"  Outlet features computed: {len(outlet_hist):,} outlets")
    print(f"  Outlets with censoring_score > 0.30: {n_constrained:,} "
          f"({100*n_constrained/len(outlet_hist):.1f}%)")

    # -----------------------------------------------------------------------
    # 6. Merge outlet meta + POI + seasonality
    # -----------------------------------------------------------------------
    print("\n[6/9] Merging feature tables...")
    gold = outlet_hist.merge(outlet, on="Outlet_ID", how="left")
    gold = gold.merge(coords, on="Outlet_ID", how="left")

    if poi_cols:
        merge_cols = ["Outlet_ID"] + poi_cols + h3_col
        gold = gold.merge(poi[merge_cols], on="Outlet_ID", how="left")
        for c in poi_cols:
            gold[c] = gold[c].fillna(0)

    # POI catchment score (weighted sum of POI types)
    if poi_cols:
        weights = {
            "poi_school":        2.0,
            "poi_bus_stop":      2.0,
            "poi_hospital":      1.5,
            "poi_tourism":       1.8,
            "poi_market":        0.5,
            "poi_place_worship": 1.2,
            "poi_fuel_station":  1.0,
            "poi_restaurant":    1.3,
            "poi_bank_atm":      1.0,
        }
        gold["poi_catchment_raw"] = sum(
            gold.get(col, pd.Series(0, index=gold.index)) * w
            for col, w in weights.items()
            if col in gold.columns
        )
        poi_max = gold["poi_catchment_raw"].quantile(0.95)
        gold["poi_catchment_score"] = (
            gold["poi_catchment_raw"] / (poi_max + 1e-9)
        ).clip(0, 1)
    else:
        gold["poi_catchment_score"] = 0.0

    # Peer efficiency gap (SFA proxy)
    print("\n[7/9] Computing peer efficiency gaps (SFA proxy)...")
    peer_p90 = (
        gold.groupby(["Outlet_Type", "Outlet_Size"])["hist_median_vol"]
        .apply(np.array)
        .to_dict()
    )

    def get_peer_gap(row):
        key   = (row["Outlet_Type"], row["Outlet_Size"])
        peers = peer_p90.get(key, np.array([]))
        if len(peers) < 5:
            return 1.0
        return peer_efficiency_gap(row["hist_median_vol"], peers)

    gold["peer_efficiency_gap"] = gold.apply(get_peer_gap, axis=1)

    # January 2026 seasonality
    jan_season = (
        season[season["Month"] == 1]
        .groupby("Distributor_ID")["Seasonality_Index"]
        .agg(lambda x: x.mode().iloc[0])
        .reset_index()
        .rename(columns={"Seasonality_Index": "jan_seasonality"})
    )
    gold = gold.merge(
        jan_season, left_on="primary_dist", right_on="Distributor_ID",
        how="left", suffixes=("", "_dist")
    )
    gold["jan_seasonality"]    = gold["jan_seasonality"].fillna("Moderate")
    gold["jan_season_factor"]  = gold["jan_seasonality"].map(SEASONALITY_MULTIPLIER).fillna(1.0)
    gold["target_season_factor"] = gold["jan_season_factor"]
    gold["target_month"]         = 1

    # Size / type factors kept as interpretability signals
    gold["size_factor"] = gold["Outlet_Size"].map(SIZE_POTENTIAL_FACTOR).fillna(1.0)
    gold["type_factor"] = gold["Outlet_Type"].map(TYPE_POTENTIAL_FACTOR).fillna(1.0)

    # Categorical encoding for LightGBM
    for cat_col in ["Outlet_Type", "Outlet_Size", "jan_seasonality"]:
        if cat_col in gold.columns:
            gold[cat_col] = gold[cat_col].astype("category")

    # -----------------------------------------------------------------------
    # 8. Train LightGBM model on historical panel
    # -----------------------------------------------------------------------
    print("\n[8/9] Training LightGBM demand model...")
    print("  Building training panel (sliding window, no leakage)...")
    train_df = build_training_records(
        monthly, outlet, season, poi, poi_cols, n_months=N_TRAINING_MONTHS
    )

    # Encode categoricals in training data
    cat_cols_model = ["Outlet_Type", "Outlet_Size", "target_seasonality"]
    for cat_col in cat_cols_model:
        if cat_col in train_df.columns:
            train_df[cat_col] = train_df[cat_col].astype("category")

    feat_cols = get_feature_cols(train_df, poi_cols)
    # Add categorical columns if not already present
    extra_cats = [c for c in cat_cols_model if c in train_df.columns and c not in feat_cols]
    feat_cols += extra_cats

    X_train = train_df[feat_cols].copy()
    y_train = train_df["target_log_vol"].values

    print(f"  Feature matrix: {X_train.shape[0]:,} rows × {X_train.shape[1]} features")
    print(f"  Features: {feat_cols}")

    # Train final model on all training data
    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(X_train, y_train,
              categorical_feature=[c for c in cat_cols_model if c in feat_cols])

    print(f"  LightGBM training complete. Best iteration: {model.best_iteration_}")

    # Feature importances
    fi = pd.Series(model.feature_importances_, index=feat_cols).sort_values(ascending=False)
    print("\n  Top-10 Feature Importances:")
    for fname, imp in fi.head(10).items():
        print(f"    {fname:<35} {imp:.0f}")

    # -----------------------------------------------------------------------
    # 9. Predict January 2026
    # -----------------------------------------------------------------------
    print("\n[9/9] Predicting Maximum Monthly Liters for January 2026...")

    # Encode categoricals and alias target_seasonality for inference
    if "jan_seasonality" in gold.columns:
        gold["target_seasonality"] = gold["jan_seasonality"].astype("category")
    for cat_col in cat_cols_model:
        if cat_col in gold.columns:
            gold[cat_col] = gold[cat_col].astype("category")

    # Align inference feature columns with training
    inf_feat_cols = [c for c in feat_cols if c in gold.columns]
    missing_feats = [c for c in feat_cols if c not in gold.columns]
    if missing_feats:
        print(f"  [WARN] Missing inference features (filling 0): {missing_feats}")
        for c in missing_feats:
            gold[c] = 0

    X_pred = gold[feat_cols].copy()
    # Fill any remaining NaN
    for c in feat_cols:
        if X_pred[c].dtype.kind == 'f':
            X_pred[c] = X_pred[c].fillna(X_pred[c].median())

    log_pred     = model.predict(X_pred)
    gold["Maximum_Monthly_Liters"] = np.clip(np.expm1(log_pred), 1.0, None).round(2)

    # --- Light censoring-aware calibration ---
    # For highly constrained outlets (censoring_score > 0.50), apply a small
    # additive uplift to ensure the model isn't systematically biased downward
    # by the censored observations in its training data.
    # The uplift is proportional to censoring_score and is capped at +15%.
    high_cens_mask = gold["censoring_score"] > 0.50
    uplift_factor  = 1.0 + (gold.loc[high_cens_mask, "censoring_score"] - 0.50) * 0.30
    gold.loc[high_cens_mask, "Maximum_Monthly_Liters"] = (
        gold.loc[high_cens_mask, "Maximum_Monthly_Liters"] * uplift_factor
    ).clip(lower=1.0).round(2)

    # Effective multiplier (for downstream visualisation compatibility)
    gold["jan_base"] = gold["jan_hist_mean"].where(
        gold["jan_hist_mean"].notna() & (gold["jan_hist_mean"] > 0),
        gold["hist_median_vol"]
    )
    gold["potential_multiplier"] = (
        gold["Maximum_Monthly_Liters"] / (gold["jan_base"] + 1e-9)
    ).clip(upper=5.0)
    gold["growth_factor"] = (1.0 + gold["yoy_growth"].clip(-0.20, 0.35))

    # -----------------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------------
    extra_cens_cols = ["cens_plateau", "cens_dist_cap", "cens_stagnation",
                       "cens_cv_score", "cens_q4_suppression"]
    behavioral_cols = ["capacity_proximity_ratio", "purchase_pace_variance", "dist_rank_mean"]

    gold_cols_to_save = [
        "Outlet_ID", "Outlet_Type", "Outlet_Size", "Cooler_Count",
        "Latitude", "Longitude",
        "hist_mean_vol", "hist_median_vol", "hist_max_vol", "hist_p75_vol", "hist_p90_vol",
        "hist_cv", "hist_months", "censoring_score", "yoy_growth",
        "jan_hist_mean", "primary_dist", "jan_seasonality",
        "size_factor", "type_factor", "jan_season_factor",
        "poi_catchment_score", "peer_efficiency_gap",
        "potential_multiplier", "growth_factor", "jan_base",
        "Maximum_Monthly_Liters",
    ] + extra_cens_cols + behavioral_cols + [c for c in poi_cols if c in gold.columns]

    if h3_col and h3_col[0] in gold.columns:
        gold_cols_to_save += h3_col

    gold_save = gold[[c for c in gold_cols_to_save if c in gold.columns]].copy()
    gold_save.to_parquet(GOLD_DIR / "gold_features.parquet", index=False)
    print(f"  Gold features saved: {GOLD_DIR / 'gold_features.parquet'}")

    predictions = gold_save[["Outlet_ID", "Maximum_Monthly_Liters"]].copy()
    predictions.to_csv(OUTPUT / "AI_ACES_predictions.csv", index=False)
    print(f"  Predictions saved:   {OUTPUT / 'AI_ACES_predictions.csv'}")

    # -----------------------------------------------------------------------
    # Censoring Analysis — decile breakdown for interpretability
    # -----------------------------------------------------------------------
    print("\n[ANALYSIS] Censoring Score Decile Breakdown:")
    gold["censoring_decile"] = pd.qcut(
        gold["censoring_score"], q=10, labels=False, duplicates="drop"
    ) + 1

    cens_analysis = (
        gold.groupby("censoring_decile")
        .agg(
            outlet_count=(            "Outlet_ID",                "count"),
            avg_censoring_score=(     "censoring_score",          "mean"),
            avg_hist_median_vol=(     "hist_median_vol",          "mean"),
            avg_predicted_vol=(       "Maximum_Monthly_Liters",   "mean"),
            avg_potential_multiplier= ("potential_multiplier",    "mean"),
            avg_capacity_proximity=(  "capacity_proximity_ratio", "mean"),
        )
        .round(3)
        .reset_index()
    )
    cens_analysis["uplift_ratio"] = (
        cens_analysis["avg_predicted_vol"] / cens_analysis["avg_hist_median_vol"]
    ).round(3)

    cens_analysis.to_csv(OUTPUT / "censoring_analysis.csv", index=False)
    print(cens_analysis.to_string(index=False))
    print(f"\n  Censoring analysis saved: {OUTPUT / 'censoring_analysis.csv'}")

    # -----------------------------------------------------------------------
    # Summary statistics
    # -----------------------------------------------------------------------
    print("\n[STATS] Prediction Summary:")
    print(predictions["Maximum_Monthly_Liters"].describe().round(2))
    print(f"\nTotal outlets predicted:        {len(predictions):,}")
    print(f"Total potential (Jan 2026):     {predictions['Maximum_Monthly_Liters'].sum():,.0f} L")
    print(f"Avg potential multiplier:       {gold['potential_multiplier'].mean():.2f}x")
    n_hi = (gold["censoring_score"] > 0.30).sum()
    print(f"High-censoring outlets (>0.30): {n_hi:,} ({100*n_hi/len(gold):.1f}%)")
    print(f"Avg capacity proximity ratio:   {gold['capacity_proximity_ratio'].mean():.3f}")
    print(f"Avg distributor rank:           {gold['dist_rank_mean'].mean():.3f}")

    print("\n[OK]  Gold layer complete.\n")


if __name__ == "__main__":
    main()
