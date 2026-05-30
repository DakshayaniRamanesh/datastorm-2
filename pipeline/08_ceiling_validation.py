"""
DataStorm 2026 - Ceiling Validation Protocol (Priority 1)
=========================================================
Validates latent potential models against CEILING proxies, not only observed sales.

Metrics per model:
  - Spearman rank vs hist_p90 / ceiling proxy
  - Log-MAE to ceiling (weighted by censoring)
  - Uplift vs censoring monotonicity
  - Gap recovery (% pred > observed)
  - High-censoring subset (score > 0.4)

Compares: Two-regime Heuristic vs LightGBM 90th-percentile benchmark.
"""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

sys.path.insert(0, str(Path(__file__).parent))

from ceiling_metrics import compute_ceiling_metrics
from feature_builder import build_outlet_features
from latent_heuristic import (
    PRIMARY_MODEL_NAME,
    compute_heuristic_potential,
    ensure_heuristic_inputs,
    finalize_latent_predictions,
    predict_latent_potential,
)

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("CeilingValidation")

ROOT = Path(__file__).parent.parent
SILVER = ROOT / "pipeline" / "silver"
GOLD = ROOT / "pipeline" / "gold"
POI_CACHE = ROOT / "pipeline" / "poi_cache"
OUTPUT = ROOT / "output"
OUTPUT.mkdir(parents=True, exist_ok=True)

SEASONALITY_MULTIPLIER = {"Favorable": 1.15, "Moderate": 1.00, "Un-Favorable": 0.88}
SIZE_POTENTIAL_FACTOR = {"Extra Large": 1.30, "Large": 1.15, "Medium": 1.00, "Small": 0.88}
TYPE_POTENTIAL_FACTOR = {
    "Grocery": 1.10, "Hotel": 1.20, "Pharmacy": 0.90, "Kiosk": 0.85,
    "Eatery": 1.05, "Bakery": 0.95, "SMMT": 1.25,
}
N_SPLITS = 4
N_TRAIN_TARGETS = 6
LGB_PARAMS = {
    "objective": "quantile", "alpha": 0.90, "metric": "quantile",
    "n_estimators": 250, "learning_rate": 0.05, "num_leaves": 31,
    "verbose": -1, "n_jobs": -1, "random_state": 42,
}


def add_dist_stats(monthly: pd.DataFrame) -> pd.DataFrame:
    dist_med = (
        monthly.groupby(["Distributor_ID", "Month"])["monthly_volume"]
        .median().rename("dist_month_median").reset_index()
    )
    monthly = monthly.merge(dist_med, on=["Distributor_ID", "Month"], how="left")
    monthly["dist_month_rank"] = monthly.groupby(["Distributor_ID", "Month"])["monthly_volume"].rank(pct=True)
    return monthly


def build_val_panel(monthly, outlet, season, poi_df, train_periods, val_periods) -> pd.DataFrame:
    """Build validation rows for each month in val_periods."""
    records = []
    for p_idx in val_periods:
        yr, mo = p_idx // 12, p_idx % 12
        if mo == 0:
            yr, mo = yr - 1, 12

        train_window = monthly[monthly["period_idx"] < p_idx].copy()
        if train_window.empty:
            continue
        train_window = train_window.drop(columns=[c for c in ["dist_month_median", "dist_month_rank"] if c in train_window.columns])
        train_window = add_dist_stats(train_window)

        feats = build_outlet_features(train_window, target_month=mo)
        feats = feats.merge(outlet[["Outlet_ID", "Outlet_Type", "Outlet_Size", "Cooler_Count"]], on="Outlet_ID", how="left")

        mo_season = (
            season[season["Month"] == mo].groupby("Distributor_ID")["Seasonality_Index"]
            .agg(lambda x: x.mode().iloc[0] if len(x) > 0 else "Moderate")
            .reset_index().rename(columns={"Seasonality_Index": "target_seasonality"})
        )
        feats = feats.merge(mo_season, left_on="primary_dist", right_on="Distributor_ID", how="left")
        feats["target_seasonality"] = feats["target_seasonality"].fillna("Moderate")
        feats["target_season_factor"] = feats["target_seasonality"].map(SEASONALITY_MULTIPLIER).fillna(1.0)
        feats["size_factor"] = feats["Outlet_Size"].map(SIZE_POTENTIAL_FACTOR).fillna(1.0)
        feats["type_factor"] = feats["Outlet_Type"].map(TYPE_POTENTIAL_FACTOR).fillna(1.0)
        feats["target_month"] = mo

        if not poi_df.empty:
            feats = feats.merge(poi_df, on="Outlet_ID", how="left")

        hist_for_peer = build_outlet_features(train_window, target_month=mo)
        hist_for_peer = hist_for_peer.merge(outlet[["Outlet_ID", "Outlet_Type", "Outlet_Size"]], on="Outlet_ID", how="left")
        peer_tbl = (
            hist_for_peer.groupby(["Outlet_Type", "Outlet_Size"])["hist_median_vol"]
            .quantile(0.90).rename("peer_p90").reset_index()
        )
        feats = feats.merge(peer_tbl, on=["Outlet_Type", "Outlet_Size"], how="left")
        feats["peer_p90"] = feats["peer_p90"].fillna(feats["hist_median_vol"])
        feats["peer_efficiency_gap"] = (feats["peer_p90"] / (feats["hist_median_vol"] + 1e-9)).clip(1.0, 3.0)
        feats = feats.drop(columns=["peer_p90"])

        feats["jan_base"] = feats.get("jan_hist_mean", feats["hist_median_vol"])
        if "jan_hist_mean" in feats.columns:
            feats["jan_base"] = feats["jan_hist_mean"].where(
                feats["jan_hist_mean"].notna() & (feats["jan_hist_mean"] > 0),
                feats["hist_median_vol"],
            )

        actual = (
            monthly[(monthly["Year"] == yr) & (monthly["Month"] == mo)]
            .groupby("Outlet_ID")["monthly_volume"].sum().rename("actual").reset_index()
        )
        feats = feats.merge(actual, on="Outlet_ID", how="inner")
        feats = feats[feats["actual"] > 0]
        feats["period_idx"] = p_idx
        records.append(feats)

    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def observed_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def validate_gold_snapshot() -> pd.DataFrame:
    """Ceiling metrics on full gold feature table (production Jan 2026 predictions)."""
    gold_path = GOLD / "gold_features.parquet"
    if not gold_path.exists():
        return pd.DataFrame()

    gold = pd.read_parquet(gold_path)
    gold = ensure_heuristic_inputs(gold)
    y_heur = gold["Maximum_Monthly_Liters"].values if "Maximum_Monthly_Liters" in gold.columns else predict_latent_potential(gold)
    y_obs = gold["hist_median_vol"].values

    rows = []
    for model_name, y_pred in [(PRIMARY_MODEL_NAME, y_heur)]:
        m = compute_ceiling_metrics(y_pred, gold, y_obs=y_obs)
        m["Model"] = model_name
        m["eval_scope"] = "gold_snapshot_jan2026"
        m["observed_mae_vs_median"] = observed_mae(y_obs, y_pred)
        rows.append(m)

    return pd.DataFrame(rows)


def walkforward_ceiling_validation() -> pd.DataFrame:
    tx = pd.read_parquet(SILVER / "transactions.parquet")
    outlet = pd.read_parquet(SILVER / "outlet_master.parquet")
    season = pd.read_parquet(SILVER / "distributor_seasonality.parquet")
    poi_df = pd.read_parquet(POI_CACHE / "poi_features.parquet") if (POI_CACHE / "poi_features.parquet").exists() else pd.DataFrame()

    monthly = (
        tx.groupby(["Outlet_ID", "Year", "Month", "Distributor_ID"])
        .agg(monthly_volume=("Volume_Liters", "sum")).reset_index()
    )
    monthly["period_idx"] = monthly["Year"] * 12 + monthly["Month"]
    periods = sorted(monthly["period_idx"].unique())

    feature_cols = [
        "hist_mean_vol", "hist_median_vol", "hist_max_vol", "hist_p75_vol", "hist_p90_vol",
        "hist_cv", "hist_months", "censoring_score", "yoy_growth",
        "target_season_factor", "target_month", "Cooler_Count",
        "combined_catchment_score", "competitor_density_gaussian", "competition_dampener",
        "peer_efficiency_gap", "size_factor", "type_factor",
    ]

    tss = TimeSeriesSplit(n_splits=N_SPLITS)
    rows = []

    for fold_idx, (train_pos, val_pos) in enumerate(tss.split(np.arange(len(periods)))):
        if len(train_pos) < 12:
            continue
        train_periods = [periods[i] for i in train_pos]
        val_periods = [periods[i] for i in val_pos]

        val_df = build_val_panel(monthly, outlet, season, poi_df, train_periods, val_periods)
        if val_df.empty or len(val_df) < 200:
            continue

        val_df = ensure_heuristic_inputs(val_df)
        y_obs = val_df["actual"].values
        y_heur = predict_latent_potential(val_df)

        for model_name, y_pred in [
            (PRIMARY_MODEL_NAME, y_heur),
        ]:
            m = compute_ceiling_metrics(y_pred, val_df, y_obs=y_obs)
            m.update({
                "Model": model_name,
                "eval_scope": "walk_forward",
                "fold": fold_idx + 1,
                "observed_mae": observed_mae(y_obs, y_pred),
                "n_outlets": len(val_df),
            })
            rows.append(m)

        # LightGBM benchmark on same fold (train on prior periods)
        train_rows = []
        for p_idx in train_periods[-N_TRAIN_TARGETS * 2:]:
            yr, mo = p_idx // 12, p_idx % 12
            if mo == 0:
                yr, mo = yr - 1, 12
            tw = monthly[monthly["period_idx"] < p_idx].copy()
            if tw.empty:
                continue
            tw = add_dist_stats(tw)
            f = build_outlet_features(tw, target_month=mo)
            f = f.merge(outlet[["Outlet_ID", "Outlet_Type", "Outlet_Size", "Cooler_Count"]], on="Outlet_ID", how="left")
            f["target_season_factor"] = 1.0
            f["size_factor"] = f["Outlet_Size"].map(SIZE_POTENTIAL_FACTOR).fillna(1.0)
            f["type_factor"] = f["Outlet_Type"].map(TYPE_POTENTIAL_FACTOR).fillna(1.0)
            if not poi_df.empty:
                f = f.merge(poi_df, on="Outlet_ID", how="left")
            act = monthly[monthly["period_idx"] == p_idx].groupby("Outlet_ID")["monthly_volume"].sum().rename("actual").reset_index()
            f = f.merge(act, on="Outlet_ID", how="inner")
            f = f[f["actual"] > 0]
            f["target_log_vol"] = np.log1p(f["actual"])
            train_rows.append(f)
        if not train_rows:
            continue
        train_panel = pd.concat(train_rows, ignore_index=True)
        cols = [c for c in feature_cols if c in train_panel.columns and c in val_df.columns]
        X_train = train_panel[cols].fillna(0.0)
        y_train = train_panel["target_log_vol"].values
        X_val = val_df[cols].fillna(0.0)

        lgb_model = lgb.LGBMRegressor(**LGB_PARAMS)
        lgb_model.fit(X_train, y_train)
        y_lgb = np.expm1(lgb_model.predict(X_val))

        m_lgb = compute_ceiling_metrics(y_lgb, val_df, y_obs=y_obs)
        m_lgb.update({
            "Model": "LightGBM Quantile (benchmark)",
            "eval_scope": "walk_forward",
            "fold": fold_idx + 1,
            "observed_mae": observed_mae(y_obs, y_lgb),
            "n_outlets": len(val_df),
        })
        rows.append(m_lgb)

        logger.info(
            "Fold %s | Heuristic ceiling_rank=%.3f uplift_cens=%.3f | LGBM ceiling_rank=%.3f observed_mae H=%.1f L=%.1f",
            fold_idx + 1,
            rows[-2]["rank_corr_ceiling_spearman"],
            rows[-2]["uplift_censoring_spearman"],
            m_lgb["rank_corr_ceiling_spearman"],
            rows[-2]["observed_mae"],
            m_lgb["observed_mae"],
        )

    return pd.DataFrame(rows)


def main() -> None:
    logger.info("Starting ceiling validation protocol...")

    snapshot = validate_gold_snapshot()
    walkforward = walkforward_ceiling_validation()

    parts = [df for df in [snapshot, walkforward] if not df.empty]
    if not parts:
        logger.error("No ceiling validation results produced.")
        sys.exit(1)

    report = pd.concat(parts, ignore_index=True)
    out_path = OUTPUT / "ceiling_validation_report.csv"
    report.to_csv(out_path, index=False)
    logger.info("Ceiling validation report saved -> %s", out_path)

    if not walkforward.empty:
        summary = walkforward.groupby("Model").agg({
            "rank_corr_ceiling_spearman": "mean",
            "uplift_censoring_spearman": "mean",
            "pct_pred_exceeds_observed": "mean",
            "observed_mae": "mean",
            "log_mae_to_ceiling_weighted": "mean",
        }).reset_index()
        logger.info("\n%s", summary.to_string(index=False))

    logger.info("Ceiling validation complete.")


if __name__ == "__main__":
    main()
