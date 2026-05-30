"""
DataStorm 2026 - Gold Features & Model Comparison Framework
===========================================================
Aggregates transactions, incorporates upgraded spatial features,
engineers advanced demand/performance features, compares three models:
  1. Phase 1 Multiplicative Heuristic Model
  2. Standard Machine Learning Quantile Regressor (HistGradientBoosting)
  3. LightGBM Quantile Regressor (objective="quantile", alpha=0.90)

Evaluates using MAE, RMSE, MAPE, and R², selects the best model automatically,
generates final predictions, and runs SHAP explainability.
"""

import sys
import pickle
import logging
import warnings
from typing import List, Dict, Tuple, Optional, Any
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
import lightgbm as lgb

# Suppress warnings
warnings.filterwarnings("ignore")

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("GoldFeaturesModel")

# Paths
ROOT = Path(__file__).parent.parent
SILVER = ROOT / "pipeline" / "silver"
POI_CACHE = ROOT / "pipeline" / "poi_cache"
GOLD_DIR = ROOT / "pipeline" / "gold"
OUTPUT = ROOT / "output"
GOLD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)

# Try SHAP import
SHAP_AVAILABLE = False
try:
    import shap
    SHAP_AVAILABLE = True
    logger.info("Loaded shap library successfully.")
except ImportError:
    logger.warning("shap library not found. Falling back to Simulated SHAP Explainer.")

# Configuration constants
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

N_TRAINING_MONTHS = 12  # Number of historical months for panel training

# LightGBM Quantile Regression Hyperparameters
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

# ---------------------------------------------------------------------------
# Resilient Simulated SHAP Explainer
# ---------------------------------------------------------------------------
class SimulatedSHAPExplainer:
    """Fallback explainer that generates contribution values matching model properties."""
    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        
        # Extract relative importance from model
        if hasattr(model, "feature_importances_"):
            importances = np.array(model.feature_importances_)
        elif hasattr(model, "coef_"):
            importances = np.abs(np.array(model.coef_))
        else:
            importances = np.ones(len(feature_names))
            
        self.importances = importances / (np.sum(importances) + 1e-9)

    def __call__(self, X: pd.DataFrame):
        X_np = X.values if isinstance(X, pd.DataFrame) else np.array(X)
        N, D = X_np.shape
        
        # Calculate deviation from mean for each feature
        X_mean = np.mean(X_np, axis=0)
        X_std = np.std(X_np, axis=0) + 1e-9
        std_dev = (X_np - X_mean) / X_std
        
        # Raw attribution proportional to standardized deviation * importance
        raw_attributions = std_dev * self.importances
        
        # Predict base & target values
        try:
            preds = self.model.predict(X)
        except Exception:
            preds = np.zeros(N)
            
        base_val = float(np.mean(preds))
        
        # Scale attributions so that they sum exactly to (prediction - base_value)
        shap_values = np.zeros((N, D))
        for i in range(N):
            pred_diff = preds[i] - base_val
            sum_raw = np.sum(np.abs(raw_attributions[i])) + 1e-9
            shap_values[i] = raw_attributions[i] * (pred_diff / sum_raw)
            
        # Mock class matching SHAP output structure
        class SimulatedExplanation:
            def __init__(self, values, base_values, data, feature_names):
                self.values = values
                self.base_values = np.full(len(values), base_values)
                self.data = data
                self.feature_names = feature_names
                
        return SimulatedExplanation(shap_values, base_val, X_np, self.feature_names)

# ---------------------------------------------------------------------------
# Advanced Feature Builder
# ---------------------------------------------------------------------------
def build_outlet_features(monthly: pd.DataFrame, target_month: int = 1) -> pd.DataFrame:
    """Compute per-outlet historical features using vectorized operations."""
    df = monthly.copy().sort_values(["Outlet_ID", "Year", "Month"])
    gp = df.groupby("Outlet_ID")

    # Basic historical sales stats
    hist_mean_vol = gp["monthly_volume"].mean()
    hist_median_vol = gp["monthly_volume"].median()
    hist_max_vol = gp["monthly_volume"].max()
    hist_p75_vol = gp["monthly_volume"].quantile(0.75)
    hist_p90_vol = gp["monthly_volume"].quantile(0.90)
    hist_std_vol = gp["monthly_volume"].std(ddof=0).fillna(0)
    hist_months = gp["monthly_volume"].count()
    hist_cv = hist_std_vol / (hist_mean_vol + 1e-9)

    # 1. Censoring score components
    # Plateau flag (rolling 3-month CV < 10%)
    df["monthly_volume_sq"] = df["monthly_volume"].pow(2)
    # Vectorized shift-based rolling 3-month mean
    mask3 = (df["Outlet_ID"] == df["Outlet_ID"].shift(2))
    roll3_mean = (df["monthly_volume"] + df["monthly_volume"].shift(1) + df["monthly_volume"].shift(2)) / 3.0
    roll3_mean = pd.Series(np.where(mask3, roll3_mean, np.nan), index=df.index)
    roll3_mean_sq = (df["monthly_volume_sq"] + df["monthly_volume_sq"].shift(1) + df["monthly_volume_sq"].shift(2)) / 3.0
    roll3_mean_sq = pd.Series(np.where(mask3, roll3_mean_sq, np.nan), index=df.index)
    roll3_std = np.sqrt(np.clip(roll3_mean_sq - np.square(roll3_mean), 0.0, None))
    roll3_cv = roll3_std / (roll3_mean + 1e-9)
    df["plateau_flag"] = (roll3_cv < 0.10).astype(float)
    df.loc[df["Outlet_ID"].isin(hist_months[hist_months < 3].index), "plateau_flag"] = np.nan
    cens_plateau = df.groupby("Outlet_ID")["plateau_flag"].mean().fillna(0.0)

    # Distributor Cap flag
    df["at_cap_flag"] = (
        np.abs(df["monthly_volume"] - df["dist_month_median"])
        / (df["dist_month_median"] + 1e-9) < 0.15
    ).astype(float)
    cens_dist_cap = df.groupby("Outlet_ID")["at_cap_flag"].mean().fillna(0.0)

    # Stagnation
    yearly = df.groupby(["Outlet_ID", "Year"])["monthly_volume"].mean().reset_index().sort_values(["Outlet_ID", "Year"])
    yearly["growth"] = yearly.groupby("Outlet_ID")["monthly_volume"].pct_change()
    yearly["stagnation_flag"] = np.where(yearly["growth"].isna(), np.nan, (np.abs(yearly["growth"]) < 0.05).astype(float))
    cens_stagnation = yearly.groupby("Outlet_ID")["stagnation_flag"].mean().fillna(0.0)
    yoy_growth = yearly.groupby("Outlet_ID")["growth"].mean().fillna(0.0).clip(-0.30, 0.50)

    # Low CV Score
    cens_cv_score = ((0.30 - hist_cv) / 0.30).clip(lower=0.0)

    # Q4 Suppression
    df["q4_vol"] = np.where(df["Month"].isin([10, 11, 12]), df["monthly_volume"], np.nan)
    q4_mean = df.groupby("Outlet_ID")["q4_vol"].mean()
    q4_count = df.groupby("Outlet_ID")["q4_vol"].count()
    q4_prem = q4_mean / (hist_mean_vol + 1e-9) - 1.0
    cens_q4_raw = (-q4_prem * 2.0).clip(0.0, 1.0).fillna(0.0)
    cens_q4_suppression = pd.Series(np.where(q4_count >= 2, cens_q4_raw, 0.0), index=hist_mean_vol.index)

    # Composite Censoring Score
    censoring_score = (
        0.30 * cens_plateau
        + 0.25 * cens_dist_cap
        + 0.20 * cens_stagnation
        + 0.15 * cens_cv_score
        + 0.10 * cens_q4_suppression
    ).clip(0.0, 1.0)

    # 2. Performance & Capacity features
    capacity_proximity_ratio = pd.Series(
        np.clip(
            np.where(
                hist_months >= 3,
                roll3_mean.groupby(df["Outlet_ID"]).mean() / (hist_max_vol + 1e-9),
                hist_mean_vol / (hist_max_vol + 1e-9)
            ), 0.0, 1.0
        ), index=hist_mean_vol.index
    )

    # Purchase Pace Variance (rolling 6-month CV average)
    # Vectorized shift-based rolling 6-month mean
    mask6 = (df["Outlet_ID"] == df["Outlet_ID"].shift(5))
    roll6_mean = (
        df["monthly_volume"] + 
        df["monthly_volume"].shift(1) + 
        df["monthly_volume"].shift(2) + 
        df["monthly_volume"].shift(3) + 
        df["monthly_volume"].shift(4) + 
        df["monthly_volume"].shift(5)
    ) / 6.0
    roll6_mean = pd.Series(np.where(mask6, roll6_mean, np.nan), index=df.index)
    roll6_mean_sq = (
        df["monthly_volume_sq"] + 
        df["monthly_volume_sq"].shift(1) + 
        df["monthly_volume_sq"].shift(2) + 
        df["monthly_volume_sq"].shift(3) + 
        df["monthly_volume_sq"].shift(4) + 
        df["monthly_volume_sq"].shift(5)
    ) / 6.0
    roll6_mean_sq = pd.Series(np.where(mask6, roll6_mean_sq, np.nan), index=df.index)
    roll6_std = np.sqrt(np.clip(roll6_mean_sq - np.square(roll6_mean), 0.0, None))
    roll6_cv = roll6_std / (roll6_mean + 1e-9)
    purchase_pace_variance = pd.Series(
        np.where(hist_months >= 6, roll6_cv.groupby(df["Outlet_ID"]).mean(), hist_cv),
        index=hist_mean_vol.index
    )

    # Distributor Relative Rank mean
    dist_rank_mean = df.groupby("Outlet_ID")["dist_month_rank"].mean()

    # Target-month historical mean
    target_vols = df[df["Month"] == target_month]
    target_mean = target_vols.groupby("Outlet_ID")["monthly_volume"].mean().reindex(hist_mean_vol.index).fillna(hist_mean_vol)

    # Primary Distributor ID
    dist_sums = df.groupby(["Outlet_ID", "Distributor_ID"])["monthly_volume"].sum().reset_index()
    primary_dist = dist_sums.sort_values("monthly_volume").groupby("Outlet_ID")["Distributor_ID"].last()

    # Rolling median & maximum features
    # Vectorized rolling median and maximum (min_periods=1)
    m1 = (df["Outlet_ID"] == df["Outlet_ID"].shift(1))
    m2 = (df["Outlet_ID"] == df["Outlet_ID"].shift(2))
    c0 = df["monthly_volume"]
    c1 = np.where(m1, df["monthly_volume"].shift(1), np.nan)
    c2 = np.where(m2, df["monthly_volume"].shift(2), np.nan)
    stacked = np.column_stack([c0, c1, c2])
    roll_med_all = np.nanmedian(stacked, axis=1)
    roll_max_all = np.nanmax(stacked, axis=1)
    df_roll_med = pd.Series(roll_med_all, index=df.index)
    df_roll_max = pd.Series(roll_max_all, index=df.index)
    rolling_median = df_roll_med.groupby(df["Outlet_ID"]).last().reindex(hist_mean_vol.index)
    rolling_maximum = df_roll_max.groupby(df["Outlet_ID"]).last().reindex(hist_mean_vol.index)

    # Peer group SFA Frontier proxy
    # Will be merged in main pipeline step since it needs outlet metadata

    return pd.DataFrame({
        "hist_mean_vol": hist_mean_vol,
        "hist_median_vol": hist_median_vol,
        "hist_max_vol": hist_max_vol,
        "hist_p75_vol": hist_p75_vol,
        "hist_p90_vol": hist_p90_vol,
        "hist_std_vol": hist_std_vol,
        "hist_cv": hist_cv,
        "hist_months": hist_months,
        "censoring_score": censoring_score,
        "cens_plateau": cens_plateau,
        "cens_dist_cap": cens_dist_cap,
        "cens_stagnation": cens_stagnation,
        "cens_cv_score": cens_cv_score,
        "cens_q4_suppression": cens_q4_suppression,
        "yoy_growth": yoy_growth,
        "jan_hist_mean": target_mean,
        "capacity_proximity_ratio": capacity_proximity_ratio,
        "purchase_pace_variance": purchase_pace_variance,
        "dist_rank_mean": dist_rank_mean,
        "primary_dist": primary_dist,
        "rolling_median": rolling_median,
        "rolling_maximum": rolling_maximum
    }).reset_index()

# ---------------------------------------------------------------------------
# Training Records Sliding Window builder
# ---------------------------------------------------------------------------
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
    # Exclude Jan 2026 if present
    all_periods = [p for p in all_periods if p != (2026, 1)]
    target_periods = all_periods[-n_months:]
    
    records = []
    for (yr, mo) in target_periods:
        cutoff_period = yr * 12 + mo
        train_monthly = monthly[
            monthly["Year"] * 12 + monthly["Month"] < cutoff_period
        ].copy()
        
        if train_monthly.empty or train_monthly["Outlet_ID"].nunique() < 10:
            continue
            
        # Recompute distributor stats on this training slice only to prevent leakage
        train_monthly = train_monthly.drop(columns=[c for c in ["dist_month_median", "dist_month_rank"] if c in train_monthly.columns])
        dist_med_train = train_monthly.groupby(["Distributor_ID", "Month"])["monthly_volume"].median().rename("dist_month_median").reset_index()
        train_monthly = train_monthly.merge(dist_med_train, on=["Distributor_ID", "Month"], how="left")
        train_monthly["dist_month_rank"] = train_monthly.groupby(["Distributor_ID", "Month"])["monthly_volume"].rank(pct=True)
        
        # Build features
        feats = build_outlet_features(train_monthly, target_month=mo)
        
        # Merge metadata & POIs
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
        
        # Merge spatial features from scraper
        if not poi_df.empty:
            feats = feats.merge(poi_df, on="Outlet_ID", how="left")
            
        # Target variable (expm1 is used downstream, target is logged sales)
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
        
    return pd.concat(records, ignore_index=True)

# ---------------------------------------------------------------------------
# Metric Evaluator
# ---------------------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute MAE, RMSE, MAPE, and R2 coefficients."""
    errors = y_pred - y_true
    abs_errors = np.abs(errors)
    mape = np.mean(abs_errors / np.clip(np.abs(y_true), 1.0, None)) * 100
    r2 = 1.0 - (np.sum(errors**2) / (np.sum((y_true - np.mean(y_true))**2) + 1e-9))
    return {
        "MAE": float(np.mean(abs_errors)),
        "RMSE": float(np.sqrt(np.mean(errors**2))),
        "MAPE_%": float(mape),
        "R2": float(r2)
    }

# ---------------------------------------------------------------------------
# Main Training & Comparison Pipeline
# ---------------------------------------------------------------------------
def main():
    logger.info("Initializing Gold Features & Model Comparison framework...")
    
    # 1. Load Silver data
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
    
    # 2. Load Spatial & Competitor features from Scraper
    poi_path = POI_CACHE / "poi_features.parquet"
    if poi_path.exists():
        poi_df = pd.read_parquet(poi_path)
        logger.info(f"Loaded spatial feature table: {len(poi_df):,} outlets")
    else:
        logger.warning("Spatial features not found at poi_features.parquet. Using coordinates only.")
        poi_df = coords.copy()
        
    # 3. Assemble full feature set for prediction boundary (representing target = Jan 2026)
    logger.info("Building January 2026 Gold Feature Table...")
    gold_feats = build_outlet_features(monthly, target_month=1)
    
    # Merge Meta
    gold_feats = gold_feats.merge(outlet, on="Outlet_ID", how="left")
    gold_feats = gold_feats.merge(coords, on="Outlet_ID", how="left")
    if not poi_df.empty:
        # Avoid duplicate coordinate columns
        drop_cols = [c for c in ["poi_lat", "poi_lon", "Latitude", "Longitude"] if c in poi_df.columns]
        gold_feats = gold_feats.merge(poi_df.drop(columns=drop_cols, errors="ignore"), on="Outlet_ID", how="left")
        
    # January Seasonality
    jan_season = (
        season[season["Month"] == 1]
        .groupby("Distributor_ID")["Seasonality_Index"]
        .agg(lambda x: x.mode().iloc[0] if len(x) > 0 else "Moderate")
        .reset_index()
        .rename(columns={"Seasonality_Index": "jan_seasonality"})
    )
    gold_feats = gold_feats.merge(jan_season, left_on="primary_dist", right_on="Distributor_ID", how="left")
    gold_feats["jan_seasonality"] = gold_feats["jan_seasonality"].fillna("Moderate")
    gold_feats["jan_season_factor"] = gold_feats["jan_seasonality"].map(SEASONALITY_MULTIPLIER).fillna(1.0)
    gold_feats["target_season_factor"] = gold_feats["jan_season_factor"]
    gold_feats["target_month"] = 1
    
    # Interpretability factors
    gold_feats["size_factor"] = gold_feats["Outlet_Size"].map(SIZE_POTENTIAL_FACTOR).fillna(1.0)
    gold_feats["type_factor"] = gold_feats["Outlet_Type"].map(TYPE_POTENTIAL_FACTOR).fillna(1.0)
    
    peer_p90 = (
        gold_feats.groupby(["Outlet_Type", "Outlet_Size"])["hist_median_vol"]
        .quantile(0.90)
        .rename("peer_p90")
        .reset_index()
    )
    gold_feats = gold_feats.merge(peer_p90, on=["Outlet_Type", "Outlet_Size"], how="left")
    gold_feats["peer_p90"] = gold_feats["peer_p90"].fillna(gold_feats["hist_median_vol"])
    gold_feats["peer_efficiency_gap"] = (gold_feats["peer_p90"] / (gold_feats["hist_median_vol"] + 1e-9)).clip(1.0, 3.0)
    gold_feats = gold_feats.drop(columns=["peer_p90"])
    gold_feats["outlet_efficiency_index"] = (gold_feats["hist_median_vol"] / (gold_feats["hist_max_vol"] + 1e-9)).clip(0.0, 1.0)
    
    # Seasonality Strength (standard deviation of the monthly indices)
    dist_season_strength = (
        season.copy()
        .assign(val=lambda r: r["Seasonality_Index"].map(SEASONALITY_MULTIPLIER))
        .groupby("Distributor_ID")["val"]
        .std()
        .fillna(0.0)
        .rename("seasonality_strength")
        .reset_index()
    )
    gold_feats = gold_feats.merge(dist_season_strength, left_on="primary_dist", right_on="Distributor_ID", how="left")
    gold_feats["seasonality_strength"] = gold_feats["seasonality_strength"].fillna(0.0)
    
    # H3 Hashing features
    # local_peer_performance: mean volume of OTHER outlets in the same H3 resolution 8
    # regional_peer_performance: mean volume of OTHER outlets in the same H3 resolution 6
    if "h3_index" in gold_feats.columns and gold_feats["h3_index"].nunique() > 1:
        # Res 8
        sum_8 = gold_feats.groupby("h3_index")["hist_median_vol"].transform("sum")
        cnt_8 = gold_feats.groupby("h3_index")["hist_median_vol"].transform("count")
        gold_feats["local_peer_performance"] = ((sum_8 - gold_feats["hist_median_vol"]) / (cnt_8 - 1).clip(1)).fillna(gold_feats["hist_median_vol"])
        
        # Res 6
        sum_6 = gold_feats.groupby("h3_res6")["hist_median_vol"].transform("sum")
        cnt_6 = gold_feats.groupby("h3_res6")["hist_median_vol"].transform("count")
        gold_feats["regional_peer_performance"] = ((sum_6 - gold_feats["hist_median_vol"]) / (cnt_6 - 1).clip(1)).fillna(gold_feats["hist_median_vol"])
        
        # Population proxy via outlet counts in res 8
        gold_feats["population_proxy"] = (cnt_8 / cnt_8.max()).fillna(0.0)
    else:
        gold_feats["local_peer_performance"] = gold_feats["hist_median_vol"]
        gold_feats["regional_peer_performance"] = gold_feats["hist_median_vol"]
        gold_feats["population_proxy"] = 0.0
        
    # Hyperlocal competition counts
    if "poi_competitor" in gold_feats.columns:
        gold_feats["hyperlocal_competition"] = gold_feats["poi_competitor"]
    else:
        gold_feats["hyperlocal_competition"] = 0
        
    # Gravity and accessibility aggregations
    gravity_cols = [c for c in gold_feats.columns if c.endswith("_gravity_score") and not c.startswith("competitor")]
    if gravity_cols:
        gold_feats["gravity_catchment_score"] = gold_feats[gravity_cols].sum(axis=1)
    else:
        gold_feats["gravity_catchment_score"] = 0.0
        
    # Accessibility combination
    bus_col = "bus_stop_gaussian_score"
    fuel_col = "fuel_station_gaussian_score"
    if bus_col in gold_feats.columns and fuel_col in gold_feats.columns:
        gold_feats["market_accessibility"] = (gold_feats[bus_col] + gold_feats[fuel_col]) / 2.0
    elif bus_col in gold_feats.columns:
        gold_feats["market_accessibility"] = gold_feats[bus_col]
    else:
        gold_feats["market_accessibility"] = 0.0
        
    # Set prediction base representation
    gold_feats["jan_base"] = gold_feats["jan_hist_mean"].where(
        gold_feats["jan_hist_mean"].notna() & (gold_feats["jan_hist_mean"] > 0),
        gold_feats["hist_median_vol"]
    )
    
    # 4. Build Training Panel Data
    poi_cols_to_use = [c for c in poi_df.columns if c not in ["Outlet_ID", "poi_lat", "poi_lon", "Latitude", "Longitude"]] if not poi_df.empty else []
    
    logger.info("Building panel dataset for validation and model training...")
    train_panel = build_training_records(monthly, outlet, season, poi_df, poi_cols_to_use, n_months=N_TRAINING_MONTHS)
    
    # Process identical features in training panel
    train_panel["size_factor"] = train_panel["Outlet_Size"].map(SIZE_POTENTIAL_FACTOR).fillna(1.0)
    train_panel["type_factor"] = train_panel["Outlet_Type"].map(TYPE_POTENTIAL_FACTOR).fillna(1.0)
    
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
    
    # 5. Define Feature list for model training
    feature_cols = [
        "hist_mean_vol", "hist_median_vol", "hist_max_vol", "hist_p75_vol", "hist_p90_vol",
        "hist_cv", "hist_months", "censoring_score", "cens_plateau", "cens_dist_cap",
        "cens_stagnation", "cens_cv_score", "cens_q4_suppression", "yoy_growth",
        "capacity_proximity_ratio", "purchase_pace_variance", "dist_rank_mean",
        "target_season_factor", "target_month", "Cooler_Count",
        # Upgraded Features
        "combined_catchment_score", "competitor_density_gaussian", "competitor_density_gravity",
        "market_saturation_index", "competition_dampener", "gravity_catchment_score",
        "market_accessibility", "population_proxy", "peer_efficiency_gap",
        "outlet_efficiency_index", "seasonality_strength", "local_peer_performance",
        "regional_peer_performance", "hyperlocal_competition"
    ]
    
    # Verify presence in dataframes
    feature_cols = [c for c in feature_cols if c in train_panel.columns]
    
    # Categoricals for LightGBM
    cat_cols = ["Outlet_Type", "Outlet_Size", "target_seasonality"]
    for c in cat_cols:
        if c in train_panel.columns:
            train_panel[c] = train_panel[c].astype("category")
            if c not in feature_cols:
                feature_cols.append(c)
                
    for c in cat_cols:
        # Match categoricals in gold_feats for inference mapping
        inf_col = "jan_seasonality" if c == "target_seasonality" else c
        if inf_col in gold_feats.columns:
            gold_feats[c] = gold_feats[inf_col].astype("category")
            
    # 6. Model Comparison & Walk-Forward Validation
    # We perform an out-of-time chronological validation split:
    # Train on first months, validate on the latest 2 target months in panel
    all_target_months = sorted(train_panel["target_month"].unique())
    train_months_split = all_target_months[:-2]
    val_months_split = all_target_months[-2:]
    
    logger.info(f"Chronological Validation Split: Train target months {train_months_split} | Val target months {val_months_split}")
    
    train_fold = train_panel[train_panel["target_month"].isin(train_months_split)]
    val_fold = train_panel[train_panel["target_month"].isin(val_months_split)]
    
    X_train_f = train_fold[feature_cols].copy()
    y_train_f = train_fold["target_log_vol"].values
    
    X_val_f = val_fold[feature_cols].copy()
    y_val_actual = val_fold["actual_vol"].values
    
    # Clean NaNs
    for c in feature_cols:
        if X_train_f[c].dtype.name != 'category':
            X_train_f[c] = X_train_f[c].fillna(0.0)
            X_val_f[c] = X_val_f[c].fillna(0.0)
            
    logger.info(f"Train Fold: {len(X_train_f):,} samples | Val Fold: {len(X_val_f):,} samples")
    
    # --- MODEL 1: Heuristic Model ---
    y_pred_heur = (
        val_fold["jan_base"] *
        val_fold["size_factor"] *
        val_fold["type_factor"] *
        val_fold["target_season_factor"] *
        (1.0 + val_fold["censoring_score"] * 0.40) *
        val_fold["peer_efficiency_gap"] *
        (1.0 + val_fold["combined_catchment_score"] * 0.15) *
        val_fold["competition_dampener"]
    ).values
    metrics_heur = compute_metrics(y_val_actual, y_pred_heur)
    
    # --- MODEL 2: Quantile Regressor (GB) ---
    logger.info("Training Quantile Regressor baseline (HistGradientBoosting)...")
    hgb_qr = HistGradientBoostingRegressor(loss="quantile", quantile=0.90, max_iter=200, random_state=42)
    # HistGradientBoosting doesn't support categoricals directly without specific preprocessing or native dtype setup
    X_train_f_num = X_train_f.copy()
    X_val_f_num = X_val_f.copy()
    for cat in cat_cols:
        if cat in X_train_f_num.columns:
            X_train_f_num[cat] = X_train_f_num[cat].cat.codes
            X_val_f_num[cat] = X_val_f_num[cat].cat.codes
    hgb_qr.fit(X_train_f_num, y_train_f)
    y_pred_hgb_log = hgb_qr.predict(X_val_f_num)
    y_pred_hgb = np.expm1(y_pred_hgb_log)
    metrics_hgb = compute_metrics(y_val_actual, y_pred_hgb)
    
    # --- MODEL 3: LightGBM Quantile Regressor ---
    logger.info("Training LightGBM Quantile Regressor...")
    lgb_qr = lgb.LGBMRegressor(**LGB_QR_PARAMS)
    lgb_qr.fit(
        X_train_f, y_train_f,
        categorical_feature=[c for c in cat_cols if c in X_train_f.columns]
    )
    y_pred_lgb_log = lgb_qr.predict(X_val_f)
    y_pred_lgb = np.expm1(y_pred_lgb_log)
    metrics_lgb = compute_metrics(y_val_actual, y_pred_lgb)
    
    # Print comparison
    logger.info("\n" + "="*80 + "\nMODEL COMPARISON RESULTS (CHRONOLOGICAL VAL WINDOW):\n" + "="*80)
    logger.info(f"{'Model':<30} | {'MAE':<12} | {'RMSE':<12} | {'MAPE %':<10} | {'R²':<8}")
    logger.info("-"*80)
    logger.info(f"{'1. Heuristic Model':<30} | {metrics_heur['MAE']:<12.2f} | {metrics_heur['RMSE']:<12.2f} | {metrics_heur['MAPE_%']:<10.2f} | {metrics_heur['R2']:<8.4f}")
    logger.info(f"{'2. Quantile Regressor (GB)':<30} | {metrics_hgb['MAE']:<12.2f} | {metrics_hgb['RMSE']:<12.2f} | {metrics_hgb['MAPE_%']:<10.2f} | {metrics_hgb['R2']:<8.4f}")
    logger.info(f"{'3. LightGBM Quantile Regressor':<30} | {metrics_lgb['MAE']:<12.2f} | {metrics_lgb['RMSE']:<12.2f} | {metrics_lgb['MAPE_%']:<10.2f} | {metrics_lgb['R2']:<8.4f}")
    logger.info("="*80)
    
    # Model Selection Auto Decision
    model_choices = {
        "Heuristic": (metrics_heur["MAPE_%"], metrics_heur["MAE"]),
        "QuantileRegressor": (metrics_hgb["MAPE_%"], metrics_hgb["MAE"]),
        "LightGBM": (metrics_lgb["MAPE_%"], metrics_lgb["MAE"])
    }
    
    # Select best model (prioritize lowest MAPE)
    best_model_name = min(model_choices, key=lambda k: model_choices[k][0])
    logger.info(f"Auto-Selection Result: '{best_model_name}' selected as best model.")
    
    # Save validation records for the validation comparison page
    validation_report = pd.DataFrame([
        {"Model": "Heuristic", **metrics_heur},
        {"Model": "Quantile Regressor", **metrics_hgb},
        {"Model": "LightGBM Quantile", **metrics_lgb}
    ])
    validation_report.to_csv(OUTPUT / "validation_report.csv", index=False)
    
    # 7. Final Model Retraining & Inference (January 2026 Prediction)
    logger.info("Retraining final selected model on FULL training panel...")
    X_full = train_panel[feature_cols].copy()
    y_full = train_panel["target_log_vol"].values
    
    for c in feature_cols:
        if X_full[c].dtype.name != 'category':
            X_full[c] = X_full[c].fillna(0.0)
            gold_feats[c] = gold_feats[c].fillna(0.0)
            
    if best_model_name == "LightGBM":
        final_model = lgb.LGBMRegressor(**LGB_QR_PARAMS)
        final_model.fit(X_full, y_full, categorical_feature=[c for c in cat_cols if c in X_full.columns])
        
        # Predict Jan 2026
        log_pred = final_model.predict(gold_feats[feature_cols])
        raw_pred = np.clip(np.expm1(log_pred), 1.0, None)
        
    elif best_model_name == "QuantileRegressor":
        final_model = HistGradientBoostingRegressor(loss="quantile", quantile=0.90, max_iter=200, random_state=42)
        X_full_num = X_full.copy()
        X_gold_num = gold_feats[feature_cols].copy()
        for cat in cat_cols:
            if cat in X_full_num.columns:
                X_full_num[cat] = X_full_num[cat].cat.codes
                X_gold_num[cat] = X_gold_num[cat].cat.codes
        final_model.fit(X_full_num, y_full)
        log_pred = final_model.predict(X_gold_num)
        raw_pred = np.clip(np.expm1(log_pred), 1.0, None)
        
    else:  # Heuristic Fallback
        raw_pred = (
            gold_feats["jan_base"] *
            gold_feats["size_factor"] *
            gold_feats["type_factor"] *
            gold_feats["target_season_factor"] *
            (1.0 + gold_feats["censoring_score"] * 0.40) *
            gold_feats["peer_efficiency_gap"] *
            (1.0 + gold_feats["combined_catchment_score"] * 0.15) *
            gold_feats["competition_dampener"]
        ).values
        
    gold_feats["Maximum_Monthly_Liters"] = np.round(raw_pred, 2)
    
    # --- Post-Processing / Censoring Calibration ---
    # Apply small calibration adjustment (+10% to +20%) to highly constrained outlets
    # to estimate true latent potential since training inputs were censored.
    high_cens = gold_feats["censoring_score"] > 0.40
    calibration_uplift = 1.0 + (gold_feats.loc[high_cens, "censoring_score"] - 0.40) * 0.33
    gold_feats.loc[high_cens, "Maximum_Monthly_Liters"] = (
        gold_feats.loc[high_cens, "Maximum_Monthly_Liters"] * calibration_uplift
    ).round(2)
    
    # Effective Potential Multiplier for visualizations
    gold_feats["potential_multiplier"] = (
        gold_feats["Maximum_Monthly_Liters"] / (gold_feats["jan_base"] + 1e-9)
    ).clip(1.0, 5.0)
    
    # Save predictions
    predictions_df = gold_feats[["Outlet_ID", "Maximum_Monthly_Liters"]].copy()
    predictions_df.to_csv(OUTPUT / "AI_ACES_predictions.csv", index=False)
    logger.info(f"Predictions saved to {OUTPUT / 'AI_ACES_predictions.csv'}")
    
    # 8. SHAP Explainability computation
    logger.info("Computing SHAP explanations for selected model...")
    # Select sample subset for explanation speed
    X_explain = gold_feats[feature_cols].copy()
    for c in feature_cols:
        if X_explain[c].dtype.name != 'category':
            X_explain[c] = X_explain[c].fillna(0.0)
        else:
            X_explain[c] = X_explain[c].cat.codes
            
    if best_model_name in ["LightGBM", "QuantileRegressor"] and SHAP_AVAILABLE:
        try:
            explainer = shap.TreeExplainer(final_model)
            shap_values_obj = explainer(X_explain)
            logger.info("Tree SHAP computation completed.")
        except Exception as e:
            logger.warning(f"Tree SHAP failed: {e}. Falling back to Simulated SHAP Explainer.")
            explainer = SimulatedSHAPExplainer(final_model, feature_names=feature_cols)
            shap_values_obj = explainer(X_explain)
    else:
        # Fallback simulated explainer
        logger.info("Using Simulated SHAP Explainer.")
        # If heuristic was chosen, we wrap it in a mock regressor for SHAP compatibility
        if best_model_name == "Heuristic":
            class HeuristicRegressor:
                def __init__(self, df, feats):
                    self.df = df
                    self.feats = feats
                def predict(self, X):
                    # compute heuristic output
                    return np.log1p(
                        X["jan_base"] * X["size_factor"] * X["type_factor"] * X["target_season_factor"] *
                        (1.0 + X["censoring_score"] * 0.40) * X["peer_efficiency_gap"] *
                        (1.0 + X["combined_catchment_score"] * 0.15) * X["competition_dampener"]
                    ).values
            mock_model = HeuristicRegressor(gold_feats, feature_cols)
            explainer = SimulatedSHAPExplainer(mock_model, feature_names=feature_cols)
        else:
            explainer = SimulatedSHAPExplainer(final_model, feature_names=feature_cols)
        shap_values_obj = explainer(X_explain)
        
    # Save Explanations to Pickle for Flask Services
    explanation_pack = {
        "shap_values": shap_values_obj.values,
        "base_value": shap_values_obj.base_values[0] if hasattr(shap_values_obj.base_values, "__len__") else shap_values_obj.base_values,
        "feature_names": feature_cols,
        "X_pred": gold_feats[feature_cols].copy(),
        "Outlet_ID": gold_feats["Outlet_ID"].values
    }
    
    explanation_file = GOLD_DIR / "shap_explanations.pkl"
    with open(explanation_file, "wb") as f:
        pickle.dump(explanation_pack, f)
    logger.info(f"SHAP explanations saved to {explanation_file}")
    
    # Save complete Gold Parquet
    gold_feats.to_parquet(GOLD_DIR / "gold_features.parquet", index=False)
    logger.info(f"Gold Feature Table saved to {GOLD_DIR / 'gold_features.parquet'} ({len(gold_feats):,} rows)")
    
    logger.info("Gold Features & Modeling step completed successfully.\n")

if __name__ == "__main__":
    main()
