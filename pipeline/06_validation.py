"""
DataStorm 2026 - Out-of-Time Walk-Forward Validation
===================================================
Executes a strict out-of-time cross-validation protocol using TimeSeriesSplit.
For each chronological split:
  1. Computes historical features on the training window only (no leakage).
  2. Evaluates the Heuristic, Quantile Regressor, and LightGBM Quantile models on identical holdout windows.
  3. Records MAE, RMSE, MAPE, and R² metrics.

Saves metrics to output/validation_report.csv and plots curves to output/validation_curves.png.
"""

import sys
import logging
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import HistGradientBoostingRegressor
import lightgbm as lgb
import matplotlib.pyplot as plt

# Ensure pipeline folder is in path for direct execution
sys.path.append(str(Path(__file__).parent))

from latent_heuristic import (
    compute_heuristic_potential,
    finalize_latent_predictions,
    ensure_heuristic_inputs,
    HIGH_CENSORING_THRESHOLD,
    PRIMARY_MODEL_NAME,
)

# Suppress warnings
warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("ValidationFramework")

# Paths
ROOT = Path(__file__).parent.parent
SILVER = ROOT / "pipeline" / "silver"
POI_CACHE = ROOT / "pipeline" / "poi_cache"
OUTPUT = ROOT / "output"
OUTPUT.mkdir(parents=True, exist_ok=True)

# Configuration constants
N_SPLITS = 5
MIN_TRAIN_MONTHS = 12
N_TRAIN_TARGETS = 6

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

LGB_PARAMS = {
    "objective": "quantile",
    "alpha": 0.90,
    "metric": "quantile",
    "n_estimators": 250,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "verbose": -1,
    "n_jobs": -1,
    "random_state": 42,
}

# ---------------------------------------------------------------------------
# Vectorized Feature Builder (imported from shared feature_builder)
# ---------------------------------------------------------------------------
from feature_builder import build_outlet_features
from holiday_features import load_silver_holidays, month_holiday_features

def add_dist_stats(monthly: pd.DataFrame) -> pd.DataFrame:
    dist_med = monthly.groupby(["Distributor_ID", "Month"])["monthly_volume"].median().rename("dist_month_median").reset_index()
    monthly = monthly.merge(dist_med, on=["Distributor_ID", "Month"], how="left")
    monthly["dist_month_rank"] = monthly.groupby(["Distributor_ID", "Month"])["monthly_volume"].rank(pct=True)
    return monthly

def build_training_panel_for_window(
    monthly: pd.DataFrame,
    outlet: pd.DataFrame,
    season: pd.DataFrame,
    poi_df: pd.DataFrame,
    train_periods: list,
    n_train_targets: int = N_TRAIN_TARGETS
) -> pd.DataFrame:
    """Prepare training panel strictly within the sliding cross-validation window."""
    sorted_periods = sorted(train_periods)
    target_periods = sorted_periods[-n_train_targets:]
    
    holidays = load_silver_holidays(SILVER / "holiday_list.parquet")
        
    records = []
    for p_idx in target_periods:
        yr, mo = p_idx // 12, p_idx % 12
        if mo == 0:
            yr, mo = yr - 1, 12
            
        train_window = monthly[monthly["period_idx"] < p_idx].copy()
        if train_window.empty or train_window["Outlet_ID"].nunique() < 5:
            continue
            
        train_window = train_window.drop(columns=[c for c in ["dist_month_median", "dist_month_rank"] if c in train_window.columns])
        train_window = add_dist_stats(train_window)
        
        feats = build_outlet_features(train_window, target_month=mo)
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
        
        # Merge POIs
        if not poi_df.empty:
            feats = feats.merge(poi_df, on="Outlet_ID", how="left")
            
        # Compute spatial interaction feature
        cc = feats.get("combined_catchment_score", pd.Series(0.0, index=feats.index))
        cd = feats.get("competitor_density_gaussian", pd.Series(0.0, index=feats.index))
        feats["catchment_to_competitor_ratio"] = (cc / (cd + 1e-9)).fillna(0.0).clip(0.0, 10.0)
            
        # Actuals
        actual = (
            monthly[monthly["period_idx"] == p_idx]
            .groupby("Outlet_ID")["monthly_volume"].sum()
            .reset_index()
            .rename(columns={"monthly_volume": "actual_vol"})
        )
        feats = feats.merge(actual, on="Outlet_ID", how="inner")
        feats = feats[feats["actual_vol"] > 0].copy()
        feats["target_log_vol"] = np.log1p(feats["actual_vol"])
        
        hstats = month_holiday_features(holidays, yr, mo)
        for col, val in hstats.items():
            feats[col] = val
        
        records.append(feats)
        
    if not records:
        return pd.DataFrame()
    return pd.concat(records, ignore_index=True)

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
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

def main():
    logger.info("Starting out-of-time walk-forward validation comparison framework...")
    
    # Load raw
    tx = pd.read_parquet(SILVER / "transactions.parquet")
    outlet = pd.read_parquet(SILVER / "outlet_master.parquet")
    season = pd.read_parquet(SILVER / "distributor_seasonality.parquet")
    
    holidays = load_silver_holidays(SILVER / "holiday_list.parquet")
        
    poi_path = POI_CACHE / "poi_features.parquet"
    poi_df = pd.read_parquet(poi_path) if poi_path.exists() else pd.DataFrame()
    
    # Monthly
    monthly = (
        tx.groupby(["Outlet_ID", "Year", "Month", "Distributor_ID"])
        .agg(monthly_volume=("Volume_Liters", "sum"))
        .reset_index()
    )
    monthly["period_idx"] = monthly["Year"] * 12 + monthly["Month"]
    
    all_periods = sorted(monthly["period_idx"].unique())
    n_periods = len(all_periods)
    logger.info(f"Loaded {n_periods} monthly intervals for CV.")
    
    # Set up walk-forward validation (TimeSeriesSplit)
    tss = TimeSeriesSplit(n_splits=N_SPLITS)
    fold_results = []
    
    for fold_idx, (train_pos, val_pos) in enumerate(tss.split(np.arange(n_periods))):
        if len(train_pos) < MIN_TRAIN_MONTHS:
            continue
            
        train_periods = [all_periods[i] for i in train_pos]
        val_periods = [all_periods[i] for i in val_pos]
        
        t_min, t_max = min(train_periods), max(train_periods)
        v_min, v_max = min(val_periods), max(val_periods)
        
        logger.info(f"Fold {fold_idx+1}: Train window {t_min//12}-{t_min%12:02d} -> {t_max//12}-{t_max%12:02d} | Val window {v_min//12}-{v_min%12:02d} -> {v_max//12}-{v_max%12:02d}")
        
        # Build training panel data
        train_panel = build_training_panel_for_window(monthly, outlet, season, poi_df, train_periods, n_train_targets=N_TRAIN_TARGETS)
        if train_panel.empty:
            continue
            
        # Reconstruct advanced features
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
        
        dist_season_strength = season.copy().assign(val=lambda r: r["Seasonality_Index"].map(SEASONALITY_MULTIPLIER)).groupby("Distributor_ID")["val"].std().fillna(0.0).rename("seasonality_strength").reset_index()
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
        train_panel["gravity_catchment_score"] = train_panel[gravity_cols_t].sum(axis=1) if gravity_cols_t else 0.0
        
        bus_col = "bus_stop_gaussian_score"
        fuel_col = "fuel_station_gaussian_score"
        if bus_col in train_panel.columns and fuel_col in train_panel.columns:
            train_panel["market_accessibility"] = (train_panel[bus_col] + train_panel[fuel_col]) / 2.0
        else:
            train_panel["market_accessibility"] = 0.0
            
        train_panel["jan_base"] = train_panel["jan_hist_mean"].where(
            train_panel["jan_hist_mean"].notna() & (train_panel["jan_hist_mean"] > 0),
            train_panel["hist_median_vol"]
        )
        
        # Features lists
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
            "jan_holiday_count", "high_holiday_month",
            "holiday_poya_count", "holiday_high_effect_count", "holiday_weighted_score",
            # Advanced Features
            "vol_max_to_median_ratio", "vol_p90_to_median_ratio", "vol_mean_to_median_ratio",
            "censoring_to_cv_ratio", "cooler_per_volume", "group_mean_vol", "group_max_vol", "group_cv",
            "catchment_to_competitor_ratio",
        ]
        feature_cols = [c for c in feature_cols if c in train_panel.columns]
        
        # Categoricals setup
        cat_cols = ["Outlet_Type", "Outlet_Size", "target_seasonality"]
        for c in cat_cols:
            if c in train_panel.columns:
                train_panel[c] = train_panel[c].astype("category")
                if c not in feature_cols:
                    feature_cols.append(c)
                    
        # Split features & targets
        X_train = train_panel[feature_cols].copy()
        y_train = train_panel["target_log_vol"].values
        
        # Fill NaNs
        for c in feature_cols:
            if X_train[c].dtype.name != 'category':
                X_train[c] = X_train[c].fillna(0.0)
                
        # Build validation set for this specific holdout POS
        # We predict on the validation months (aggregated volume)
        val_monthly = monthly[monthly["period_idx"].isin(val_periods)].copy()
        val_monthly = add_dist_stats(val_monthly)
        
        # We need to construct prediction features at the validation boundary
        # i.e., features are computed using the train_periods
        train_monthly = monthly[monthly["period_idx"].isin(train_periods)].copy()
        train_monthly = add_dist_stats(train_monthly)
        
        for p_val in sorted(val_periods):
            yr, mo = p_val // 12, p_val % 12
            if mo == 0:
                yr, mo = yr - 1, 12
                
            val_feats = build_outlet_features(train_monthly, target_month=mo)
            val_feats = val_feats.merge(outlet, on="Outlet_ID", how="left")
            
            # Seasonality
            mo_season_val = (
                season[season["Month"] == mo]
                .groupby("Distributor_ID")["Seasonality_Index"]
                .agg(lambda x: x.mode().iloc[0] if len(x) > 0 else "Moderate")
                .reset_index()
                .rename(columns={"Seasonality_Index": "target_seasonality"})
            )
            val_feats = val_feats.merge(mo_season_val, left_on="primary_dist", right_on="Distributor_ID", how="left")
            val_feats["target_seasonality"] = val_feats["target_seasonality"].fillna("Moderate")
            val_feats["target_season_factor"] = val_feats["target_seasonality"].map(SEASONALITY_MULTIPLIER).fillna(1.0)
            val_feats["target_month"] = mo
            
            # Map Meta factors
            val_feats["size_factor"] = val_feats["Outlet_Size"].map(SIZE_POTENTIAL_FACTOR).fillna(1.0)
            val_feats["type_factor"] = val_feats["Outlet_Type"].map(TYPE_POTENTIAL_FACTOR).fillna(1.0)
            
            hstats = month_holiday_features(holidays, yr, mo)
            for col, val in hstats.items():
                val_feats[col] = val
            
            if not poi_df.empty:
                drop_cols = [c for c in ["poi_lat", "poi_lon", "Latitude", "Longitude"] if c in poi_df.columns]
                val_feats = val_feats.merge(poi_df.drop(columns=drop_cols, errors="ignore"), on="Outlet_ID", how="left")
                
            # Compute spatial interaction feature
            cc = val_feats.get("combined_catchment_score", pd.Series(0.0, index=val_feats.index))
            cd = val_feats.get("competitor_density_gaussian", pd.Series(0.0, index=val_feats.index))
            val_feats["catchment_to_competitor_ratio"] = (cc / (cd + 1e-9)).fillna(0.0).clip(0.0, 10.0)
                
            # peer gap
            peer_p90_val = (
                val_feats.groupby(["Outlet_Type", "Outlet_Size"])["hist_median_vol"]
                .quantile(0.90)
                .rename("peer_p90_val")
                .reset_index()
            )
            val_feats = val_feats.merge(peer_p90_val, on=["Outlet_Type", "Outlet_Size"], how="left")
            val_feats["peer_p90_val"] = val_feats["peer_p90_val"].fillna(val_feats["hist_median_vol"])
            val_feats["peer_efficiency_gap"] = (val_feats["peer_p90_val"] / (val_feats["hist_median_vol"] + 1e-9)).clip(1.0, 3.0)
            val_feats = val_feats.drop(columns=["peer_p90_val"])
            val_feats["outlet_efficiency_index"] = (val_feats["hist_median_vol"] / (val_feats["hist_max_vol"] + 1e-9)).clip(0.0, 1.0)
            
            val_feats = val_feats.merge(dist_season_strength, left_on="primary_dist", right_on="Distributor_ID", how="left")
            val_feats["seasonality_strength"] = val_feats["seasonality_strength"].fillna(0.0)
            
            if "h3_index" in val_feats.columns and val_feats["h3_index"].nunique() > 1:
                sum_8_v = val_feats.groupby("h3_index")["hist_median_vol"].transform("sum")
                cnt_8_v = val_feats.groupby("h3_index")["hist_median_vol"].transform("count")
                val_feats["local_peer_performance"] = ((sum_8_v - val_feats["hist_median_vol"]) / (cnt_8_v - 1).clip(1)).fillna(val_feats["hist_median_vol"])
                sum_6_v = val_feats.groupby("h3_res6")["hist_median_vol"].transform("sum")
                cnt_6_v = val_feats.groupby("h3_res6")["hist_median_vol"].transform("count")
                val_feats["regional_peer_performance"] = ((sum_6_v - val_feats["hist_median_vol"]) / (cnt_6_v - 1).clip(1)).fillna(val_feats["hist_median_vol"])
                val_feats["population_proxy"] = (cnt_8_v / cnt_8_v.max()).fillna(0.0)
            else:
                val_feats["local_peer_performance"] = val_feats["hist_median_vol"]
                val_feats["regional_peer_performance"] = val_feats["hist_median_vol"]
                val_feats["population_proxy"] = 0.0
                
            if "poi_competitor" in val_feats.columns:
                val_feats["hyperlocal_competition"] = val_feats["poi_competitor"]
            else:
                val_feats["hyperlocal_competition"] = 0
                
            val_feats["gravity_catchment_score"] = val_feats[gravity_cols_t].sum(axis=1) if gravity_cols_t else 0.0
            
            if bus_col in val_feats.columns and fuel_col in val_feats.columns:
                val_feats["market_accessibility"] = (val_feats[bus_col] + val_feats[fuel_col]) / 2.0
            else:
                val_feats["market_accessibility"] = 0.0
                
            val_feats["jan_base"] = val_feats["jan_hist_mean"].where(
                val_feats["jan_hist_mean"].notna() & (val_feats["jan_hist_mean"] > 0),
                val_feats["hist_median_vol"]
            )

            if "combined_catchment_score" not in val_feats.columns:
                val_feats["combined_catchment_score"] = 0.0
            else:
                val_feats["combined_catchment_score"] = val_feats["combined_catchment_score"].fillna(0.0)
            if "competition_dampener" not in val_feats.columns:
                val_feats["competition_dampener"] = 1.0
            else:
                val_feats["competition_dampener"] = val_feats["competition_dampener"].fillna(1.0)
            
            actual_val = (
                monthly[monthly["period_idx"] == p_val]
                .groupby("Outlet_ID")["monthly_volume"].sum()
                .reset_index()
                .rename(columns={"monthly_volume": "actual"})
            )
            
            val_combined = val_feats.merge(actual_val, on="Outlet_ID", how="inner")
            val_combined = val_combined[val_combined["actual"] > 0].copy()
            
            if val_combined.empty:
                continue
                
            # Align categoricals
            for c in cat_cols:
                if c in val_combined.columns:
                    val_combined[c] = val_combined[c].astype("category")
                    
            X_val = val_combined[feature_cols].copy()
            for c in feature_cols:
                if X_val[c].dtype.name != 'category':
                    X_val[c] = X_val[c].fillna(0.0)
                    
            # 1. Primary latent heuristic (production model)
            val_combined = ensure_heuristic_inputs(val_combined)
            y_pred_heur = finalize_latent_predictions(
                compute_heuristic_potential(val_combined), val_combined
            )
            y_actual = val_combined["actual"].values
            metrics_heur = compute_metrics(y_actual, y_pred_heur)

            # High-censoring outlets: where latent uplift claim matters most
            high_cens_mask = val_combined["censoring_score"].values > HIGH_CENSORING_THRESHOLD
            if np.sum(high_cens_mask) >= 30:
                metrics_heur_hi = compute_metrics(y_actual[high_cens_mask], y_pred_heur[high_cens_mask])
                fold_results.append({
                    "fold": fold_idx + 1,
                    "period": f"{yr}-{mo:02d}",
                    "Model": f"{PRIMARY_MODEL_NAME} (censoring>0.4)",
                    **metrics_heur_hi,
                })
            
            # 2. Evaluate Quantile Regressor
            hgb_qr = HistGradientBoostingRegressor(loss="quantile", quantile=0.90, max_iter=200, random_state=42)
            X_train_num = X_train.copy()
            X_val_num = X_val.copy()
            for cat in cat_cols:
                if cat in X_train_num.columns:
                    X_train_num[cat] = X_train_num[cat].cat.codes
                    X_val_num[cat] = X_val_num[cat].cat.codes
            hgb_qr.fit(X_train_num, y_train)
            y_pred_hgb_log = hgb_qr.predict(X_val_num)
            y_pred_hgb = np.expm1(y_pred_hgb_log)
            metrics_hgb = compute_metrics(val_combined["actual"].values, y_pred_hgb)
            
            # 3. Evaluate LightGBM
            lgb_qr = lgb.LGBMRegressor(**LGB_PARAMS)
            lgb_qr.fit(X_train, y_train, categorical_feature=[c for c in cat_cols if c in X_train.columns])
            y_pred_lgb_log = lgb_qr.predict(X_val)
            y_pred_lgb = np.expm1(y_pred_lgb_log)
            metrics_lgb = compute_metrics(val_combined["actual"].values, y_pred_lgb)
            
            # Record
            fold_results.append({"fold": fold_idx+1, "period": f"{yr}-{mo:02d}", "Model": PRIMARY_MODEL_NAME, **metrics_heur})
            fold_results.append({"fold": fold_idx+1, "period": f"{yr}-{mo:02d}", "Model": "Quantile Regressor", **metrics_hgb})
            fold_results.append({"fold": fold_idx+1, "period": f"{yr}-{mo:02d}", "Model": "LightGBM Quantile", **metrics_lgb})
            
    # Save Report
    cv_report = pd.DataFrame(fold_results)
    cv_report.to_csv(OUTPUT / "validation_report.csv", index=False)
    logger.info(f"Walk-forward validation report saved to {OUTPUT / 'validation_report.csv'}")
    
    # Compute averages
    summary = cv_report.groupby(["Model"]).agg({
        "MAE": "mean",
        "RMSE": "mean",
        "MAPE_%": "mean",
        "R2": "mean"
    }).reset_index()
    
    logger.info("\n" + "="*80 + "\nCROSS-VALIDATION AVERAGES OVER ALL FOLDS:\n" + "="*80)
    logger.info(f"{'Model':<30} | {'Mean MAE':<12} | {'Mean RMSE':<12} | {'Mean MAPE %':<12} | {'Mean R²':<8}")
    logger.info("-"*80)
    for _, r in summary.iterrows():
        logger.info(f"{r['Model']:<30} | {r['MAE']:<12.2f} | {r['RMSE']:<12.2f} | {r['MAPE_%']:<12.2f} | {r['R2']:<8.4f}")
    logger.info("="*80)
    
    # 9. Plot Validation Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    models = [PRIMARY_MODEL_NAME, "Quantile Regressor", "LightGBM Quantile"]
    colors = ["#1f77b4", "#2ca02c", "#d62728"]
    
    for model, clr in zip(models, colors):
        sub = cv_report[cv_report["Model"] == model].sort_values("fold")
        folds = sub["fold"].values
        
        ax1.plot(folds, sub["MAPE_%"], marker="o", color=clr, label=model, lw=2)
        ax2.plot(folds, sub["MAE"], marker="s", color=clr, label=model, lw=2)
        
    ax1.set_title("Walk-Forward Validation: MAPE Curve", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Chronological Fold Index", fontsize=9)
    ax1.set_ylabel("MAPE (%)", fontsize=9)
    ax1.set_xticks(np.arange(1, N_SPLITS + 1))
    ax1.legend()
    
    ax2.set_title("Walk-Forward Validation: MAE Curve", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Chronological Fold Index", fontsize=9)
    ax2.set_ylabel("MAE (Liters)", fontsize=9)
    ax2.set_xticks(np.arange(1, N_SPLITS + 1))
    ax2.legend()
    
    fig.suptitle("DataStorm 2026 - Model Validation Performance Comparisons", fontsize=13, fontweight="bold")
    plt.tight_layout()
    
    curve_path = OUTPUT / "validation_curves.png"
    fig.savefig(curve_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Validation performance curves plotted to {curve_path}")
    logger.info("Out-of-time validation framework completed successfully.\n")

if __name__ == "__main__":
    main()
