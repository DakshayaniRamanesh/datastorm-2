"""
DataStorm 2026 - Latent Heuristic Calibration
============================================
Calibrates the latent heuristic uplift coefficients (CENSORING_UPLIFT and CATCHMENT_UPLIFT)
empirically against LightGBM 90th-percentile quantile regression predictions.
"""

import sys
import pickle
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import lightgbm as lgb

# Ensure pipeline folder is in path
ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT / "pipeline"))

from feature_builder import build_outlet_features

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("HeuristicCalibration")

SILVER = ROOT / "pipeline" / "silver"
POI_CACHE = ROOT / "pipeline" / "poi_cache"
GOLD_DIR = ROOT / "pipeline" / "gold"
OUTPUT = ROOT / "output"

SEASONALITY_MULTIPLIER = {
    "Favorable": 1.15,
    "Moderate": 1.00,
    "Un-Favorable": 0.88,
}

SIZE_POTENTIAL_FACTOR = {
    "Extra Large": 1.30,
    "Large": 1.15,
    "Medium": 1.00,
    "Small": 0.88,
}

TYPE_POTENTIAL_FACTOR = {
    "Grocery": 1.10,
    "Hotel": 1.20,
    "Pharmacy": 0.90,
    "Kiosk": 0.85,
    "Eatery": 1.05,
    "Bakery": 0.95,
    "SMMT": 1.25,
}

N_TRAINING_MONTHS = 12

LGB_QR_PARAMS = {
    "objective": "quantile",
    "alpha": 0.90,
    "metric": "quantile",
    "n_estimators": 500,
    "learning_rate": 0.04,
    "num_leaves": 45,
    "max_depth": -1,
    "min_child_samples": 15,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.2,
    "verbose": -1,
    "n_jobs": -1,
    "random_state": 42,
}

def build_training_records(
    monthly: pd.DataFrame,
    outlet: pd.DataFrame,
    season: pd.DataFrame,
    poi_df: pd.DataFrame,
    poi_cols: list,
    n_months: int = N_TRAINING_MONTHS
) -> pd.DataFrame:
    """Build panel training dataset of (outlet, target_month) records strictly preventing leakage."""
    all_periods = sorted(
        monthly[["Year", "Month"]].drop_duplicates()
        .apply(lambda r: (int(r["Year"]), int(r["Month"])), axis=1)
        .tolist()
    )
    all_periods = [p for p in all_periods if p != (2026, 1)]
    target_periods = all_periods[-n_months:]
    
    # Load holiday calendar for Priority 4
    holiday_path = SILVER / "holiday_list.parquet"
    holidays = None
    if holiday_path.exists():
        holidays = pd.read_parquet(holiday_path)
        holidays["Date"] = pd.to_datetime(holidays["Date"])
        holidays["Year"] = holidays["Date"].dt.year
        holidays["Month"] = holidays["Date"].dt.month
    
    records = []
    for (yr, mo) in target_periods:
        cutoff_period = yr * 12 + mo
        train_monthly = monthly[
            monthly["Year"] * 12 + monthly["Month"] < cutoff_period
        ].copy()
        
        if train_monthly.empty or train_monthly["Outlet_ID"].nunique() < 10:
            continue
            
        train_monthly = train_monthly.drop(columns=[c for c in ["dist_month_median", "dist_month_rank"] if c in train_monthly.columns])
        dist_med_train = train_monthly.groupby(["Distributor_ID", "Month"])["monthly_volume"].median().rename("dist_month_median").reset_index()
        train_monthly = train_monthly.merge(dist_med_train, on=["Distributor_ID", "Month"], how="left")
        train_monthly["dist_month_rank"] = train_monthly.groupby(["Distributor_ID", "Month"])["monthly_volume"].rank(pct=True)
        
        # Build features
        feats = build_outlet_features(train_monthly, target_month=mo)
        
        # Merge metadata
        feats = feats.merge(outlet[["Outlet_ID", "Outlet_Type", "Outlet_Size", "Cooler_Count"]], on="Outlet_ID", how="left")
        
        # Seasonality
        mo_season = (
            season[season["Month"] == mo]
            .groupby("Distributor_ID")["Seasonality_Index"]
            .agg(lambda x: x.mode().iloc[0] if len(x) > 0 else "Moderate")
            .reset_index()
            .rename(columns={"Seasonality_Index": "target_seasonality"})
        )
        feats = feats.merge(mo_season, left_on="primary_dist", right_on="Distributor_ID", how="left")
        feats["target_seasonality"] = feats["target_seasonality"].fillna("Moderate")
        feats["target_season_factor"] = feats["target_seasonality"].map(SEASONALITY_MULTIPLIER).fillna(1.0)
        feats["target_month"] = mo
        
        # Merge spatial features
        if not poi_df.empty:
            feats = feats.merge(poi_df, on="Outlet_ID", how="left")
            
        # Target variable
        actual = (
            monthly[(monthly["Year"] == yr) & (monthly["Month"] == mo)]
            .groupby("Outlet_ID")["monthly_volume"].sum()
            .reset_index()
            .rename(columns={"monthly_volume": "actual_vol"})
        )
        feats = feats.merge(actual, on="Outlet_ID", how="inner")
        feats = feats[feats["actual_vol"] > 0].copy()
        feats["target_log_vol"] = np.log1p(feats["actual_vol"])
        
        # Holiday features (Priority 4)
        if holidays is not None:
            target_holidays = holidays[(holidays["Year"] == yr) & (holidays["Month"] == mo)]
            holiday_count = len(target_holidays)
        else:
            holiday_count = 0
        feats["jan_holiday_count"] = holiday_count
        feats["high_holiday_month"] = int(holiday_count >= 3)
        
        records.append(feats)
        
    return pd.concat(records, ignore_index=True)

def heuristic_with_params(df, alpha, gamma):
    """Computes raw heuristic potential with specified alpha and gamma."""
    # Handle ensure heuristic inputs locally for clean minimization
    d = df.copy()
    defaults = {
        "jan_base": d.get("hist_median_vol", pd.Series(0.0, index=d.index)),
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
        if col not in d.columns:
            if isinstance(default, pd.Series):
                d[col] = default.reindex(d.index).fillna(0.0)
            else:
                d[col] = default
        else:
            if col in ("competition_dampener", "peer_efficiency_gap", "size_factor", "type_factor", "target_season_factor"):
                d[col] = d[col].fillna(default)
            else:
                d[col] = pd.to_numeric(d[col], errors="coerce").fillna(0.0)
    
    d["competition_dampener"] = d["competition_dampener"].fillna(1.0).clip(0.5, 1.0)
    d["peer_efficiency_gap"] = d["peer_efficiency_gap"].fillna(1.0).clip(1.0, 3.0)
    d["combined_catchment_score"] = d["combined_catchment_score"].fillna(0.0).clip(0.0, 1.0)
    d["censoring_score"] = d["censoring_score"].fillna(0.0).clip(0.0, 1.0)
    
    return (
        d["jan_base"].values
        * d["size_factor"].values
        * d["type_factor"].values
        * d["target_season_factor"].values
        * (1.0 + d["censoring_score"].values * alpha)
        * d["peer_efficiency_gap"].values
        * (1.0 + d["combined_catchment_score"].values * gamma)
        * d["competition_dampener"].values
    )

def calibration_loss(params, df, lgb_ceiling):
    """Computes log-MSE loss between heuristic potential and LightGBM ceiling."""
    alpha, gamma = params
    # Impose boundaries
    if alpha < 0.0 or gamma < 0.0 or alpha > 1.5 or gamma > 1.0:
        return 1e9
    pred = heuristic_with_params(df, alpha, gamma)
    
    # Floor: no zero-potential outlets with any history
    median = df["hist_median_vol"].values
    floor = np.maximum(1.0, median * 0.05)
    pred = np.maximum(pred, floor)

    # Hard cap vs historical baseline (latent cannot exceed 5x jan_base)
    base = np.maximum(df["jan_base"].values, median)
    cap = base * 5.0
    pred = np.minimum(pred, cap)

    # High-censoring additional uplift
    cens = df["censoring_score"].values
    high = cens > 0.40
    if np.any(high):
        uplift = 1.0 + (cens[high] - 0.40) * 0.33
        pred[high] = pred[high] * uplift
        pred = np.minimum(pred, cap)
        
    return np.mean((np.log1p(pred) - np.log1p(lgb_ceiling))**2)

def main():
    logger.info("Loading silver transactional and metadata tables...")
    tx = pd.read_parquet(SILVER / "transactions.parquet")
    outlet = pd.read_parquet(SILVER / "outlet_master.parquet")
    coords = pd.read_parquet(SILVER / "outlet_coordinates.parquet")
    season = pd.read_parquet(SILVER / "distributor_seasonality.parquet")
    
    # Aggregation
    monthly = (
        tx.groupby(["Outlet_ID", "Year", "Month", "Distributor_ID"])
        .agg(monthly_volume=("Volume_Liters", "sum"))
        .reset_index()
    )
    dist_month_median = (
        monthly.groupby(["Distributor_ID", "Month"])["monthly_volume"]
        .median()
        .rename("dist_month_median")
        .reset_index()
    )
    monthly = monthly.merge(dist_month_median, on=["Distributor_ID", "Month"], how="left")
    monthly["dist_month_rank"] = monthly.groupby(["Distributor_ID", "Month"])["monthly_volume"].rank(pct=True)
    
    # Load spatial features
    poi_path = POI_CACHE / "poi_features.parquet"
    if poi_path.exists():
        poi_df = pd.read_parquet(poi_path)
    else:
        poi_df = coords.copy()
        
    # Build Panel
    logger.info("Assembling 12-month panel dataset...")
    poi_cols_to_use = [c for c in poi_df.columns if c not in ["Outlet_ID", "poi_lat", "poi_lon", "Latitude", "Longitude"]] if not poi_df.empty else []
    train_panel = build_training_records(monthly, outlet, season, poi_df, poi_cols_to_use, n_months=N_TRAINING_MONTHS)
    
    # Add features identically
    train_panel["size_factor"] = train_panel["Outlet_Size"].map(SIZE_POTENTIAL_FACTOR).fillna(1.0)
    train_panel["type_factor"] = train_panel["Outlet_Type"].map(TYPE_POTENTIAL_FACTOR).fillna(1.0)
    
    dist_season_strength = (
        season.copy()
        .assign(val=lambda r: r["Seasonality_Index"].map(SEASONALITY_MULTIPLIER))
        .groupby("Distributor_ID")["val"]
        .std()
        .fillna(0.0)
        .rename("seasonality_strength")
        .reset_index()
    )
    
    peer_p90_t = (
        train_panel.groupby(["Outlet_Type", "Outlet_Size", "target_month"])["hist_median_vol"]
        .quantile(0.90)
        .rename("peer_p90_t")
        .reset_index()
    )
    train_panel = train_panel.merge(peer_p90_t, on=["Outlet_Type", "Outlet_Size", "target_month"], how="left")
    train_panel["peer_p90_t"] = train_panel["peer_p90_t"].fillna(train_panel["hist_median_vol"])
    train_panel["peer_efficiency_gap"] = (train_panel["peer_p90_t"] / (train_panel["hist_median_vol"] + 1e-9)).clip(1.0, 3.0)
    train_panel = train_panel.drop(columns=["peer_p90_t"])
    train_panel["outlet_efficiency_index"] = (train_panel["hist_median_vol"] / (train_panel["hist_max_vol"] + 1e-9)).clip(0.0, 1.0)
    
    train_panel = train_panel.merge(dist_season_strength, left_on="primary_dist", right_on="Distributor_ID", how="left")
    train_panel["seasonality_strength"] = train_panel["seasonality_strength"].fillna(0.0)
    
    if "h3_index" in train_panel.columns and train_panel["h3_index"].nunique() > 1:
        sum_8_t = train_panel.groupby(["h3_index", "target_month"])["hist_median_vol"].transform("sum")
        cnt_8_t = train_panel.groupby(["h3_index", "target_month"])["hist_median_vol"].transform("count")
        train_panel["local_peer_performance"] = ((sum_8_t - train_panel["hist_median_vol"]) / (cnt_8_t - 1).clip(1)).fillna(train_panel["hist_median_vol"])
        
        sum_6_t = train_panel.groupby(["h3_res6", "target_month"])["hist_median_vol"].transform("sum")
        cnt_6_t = train_panel.groupby(["h3_res6", "target_month"])["hist_median_vol"].transform("count")
        train_panel["regional_peer_performance"] = ((sum_6_t - train_panel["hist_median_vol"]) / (cnt_6_t - 1).clip(1)).fillna(train_panel["hist_median_vol"])
        train_panel["population_proxy"] = (cnt_8_t / cnt_8_t.max()).fillna(0.0)
    else:
        train_panel["local_peer_performance"] = train_panel["hist_median_vol"]
        train_panel["regional_peer_performance"] = train_panel["hist_median_vol"]
        train_panel["population_proxy"] = 0.0
        
    if "poi_competitor" in train_panel.columns:
        train_panel["hyperlocal_competition"] = train_panel["poi_competitor"]
    else:
        train_panel["hyperlocal_competition"] = 0
        
    gravity_cols_t = [c for c in train_panel.columns if c.endswith("_gravity_score") and not c.startswith("competitor")]
    if gravity_cols_t:
        train_panel["gravity_catchment_score"] = train_panel[gravity_cols_t].sum(axis=1)
    else:
        train_panel["gravity_catchment_score"] = 0.0
        
    bus_col = "bus_stop_gaussian_score"
    fuel_col = "fuel_station_gaussian_score"
    if bus_col in train_panel.columns and fuel_col in train_panel.columns:
        train_panel["market_accessibility"] = (train_panel[bus_col] + train_panel[fuel_col]) / 2.0
    elif bus_col in train_panel.columns:
        train_panel["market_accessibility"] = train_panel[bus_col]
    else:
        train_panel["market_accessibility"] = 0.0
        
    train_panel["jan_base"] = train_panel["jan_hist_mean"].where(
        train_panel["jan_hist_mean"].notna() & (train_panel["jan_hist_mean"] > 0),
        train_panel["hist_median_vol"]
    )
    
    # LightGBM training features
    feature_cols = [
        "hist_mean_vol", "hist_median_vol", "hist_max_vol", "hist_p75_vol", "hist_p90_vol",
        "hist_cv", "hist_months", "censoring_score", "cens_plateau", "cens_dist_cap",
        "cens_stagnation", "cens_cv_score", "cens_q4_suppression", "yoy_growth",
        "capacity_proximity_ratio", "purchase_pace_variance", "dist_rank_mean",
        "target_season_factor", "target_month", "Cooler_Count",
        "combined_catchment_score", "competitor_density_gaussian", "competitor_density_gravity",
        "market_saturation_index", "competition_dampener", "gravity_catchment_score",
        "market_accessibility", "population_proxy", "peer_efficiency_gap",
        "outlet_efficiency_index", "seasonality_strength", "local_peer_performance",
        "regional_peer_performance", "hyperlocal_competition",
        "jan_holiday_count", "high_holiday_month"
    ]
    
    feature_cols = [c for c in feature_cols if c in train_panel.columns]
    
    cat_cols = ["Outlet_Type", "Outlet_Size", "target_seasonality"]
    for c in cat_cols:
        if c in train_panel.columns:
            train_panel[c] = train_panel[c].astype("category")
            if c not in feature_cols:
                feature_cols.append(c)
                
    X_full = train_panel[feature_cols].copy()
    y_full = train_panel["target_log_vol"].values
    
    for c in feature_cols:
        if X_full[c].dtype.name != "category":
            X_full[c] = X_full[c].fillna(0.0)
            
    logger.info("Training LightGBM 90th-percentile quantile model...")
    lgb_qr = lgb.LGBMRegressor(**LGB_QR_PARAMS)
    lgb_qr.fit(X_full, y_full, categorical_feature=[c for c in cat_cols if c in X_full.columns])
    
    y_pred_log = lgb_qr.predict(X_full)
    lgb_ceiling = np.expm1(y_pred_log)
    
    logger.info("Calibrating Heuristic uplift coefficients (CENSORING_UPLIFT, CATCHMENT_UPLIFT)...")
    # Perform optimization
    res = minimize(
        calibration_loss,
        x0=[0.40, 0.15],
        args=(train_panel, lgb_ceiling),
        method="Nelder-Mead"
    )
    
    alpha_cal, gamma_cal = res.x
    logger.info(f"Calibration successful! Optimized parameters:")
    logger.info(f"  CENSORING_UPLIFT (alpha) = {alpha_cal:.4f} (Original: 0.40)")
    logger.info(f"  CATCHMENT_UPLIFT (gamma) = {gamma_cal:.4f} (Original: 0.15)")
    logger.info(f"  Optimization status: {res.message}")
    
    # Save the calibrated coefficients back to latent_heuristic.py
    heuristic_file = ROOT / "pipeline" / "latent_heuristic.py"
    if heuristic_file.exists():
        content = heuristic_file.read_text(encoding="utf-8")
        
        # Replace line defining CENSORING_UPLIFT
        import re
        content_new = re.sub(
            r"CENSORING_UPLIFT\s*=\s*[0-9.]+",
            f"CENSORING_UPLIFT = {alpha_cal:.4f}",
            content
        )
        content_new = re.sub(
            r"CATCHMENT_UPLIFT\s*=\s*[0-9.]+",
            f"CATCHMENT_UPLIFT = {gamma_cal:.4f}",
            content_new
        )
        
        heuristic_file.write_text(content_new, encoding="utf-8")
        logger.info(f"Updated {heuristic_file} with calibrated parameters.")
    else:
        logger.warning(f"latent_heuristic.py not found at {heuristic_file}!")

if __name__ == "__main__":
    main()
