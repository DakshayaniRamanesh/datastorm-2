"""
DataStorm 2026 - Walk-Forward Heuristic Calibration (Ceiling Targets)
=====================================================================
Calibrates CENSORING_UPLIFT and CATCHMENT_UPLIFT using walk-forward folds
against ceiling proxies (hist_p90, hist_max) — NOT LightGBM circular fitting.

Saves:
  pipeline/gold/heuristic_calibration.json
Updates constants in pipeline/latent_heuristic.py
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import TimeSeriesSplit

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

from feature_builder import build_outlet_features
from poi_catchment import enrich_catchment_features
from latent_heuristic import (
    CATCHMENT_UPLIFT,
    CENSORING_UPLIFT,
    predict_latent_potential,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("HeuristicCalibration")

SILVER = ROOT / "pipeline" / "silver"
POI_CACHE = ROOT / "pipeline" / "poi_cache"
GOLD_DIR = ROOT / "pipeline" / "gold"
OUTPUT = ROOT / "output"

SEASONALITY_MULTIPLIER = {"Favorable": 1.15, "Moderate": 1.00, "Un-Favorable": 0.88}
SIZE_POTENTIAL_FACTOR = {"Extra Large": 1.30, "Large": 1.15, "Medium": 1.00, "Small": 0.88}
TYPE_POTENTIAL_FACTOR = {
    "Grocery": 1.10, "Hotel": 1.20, "Pharmacy": 0.90, "Kiosk": 0.85,
    "Eatery": 1.05, "Bakery": 0.95, "SMMT": 1.25,
}
N_TARGET_MONTHS = 12
N_SPLITS = 4


def add_dist_stats(monthly: pd.DataFrame) -> pd.DataFrame:
    dist_med = (
        monthly.groupby(["Distributor_ID", "Month"])["monthly_volume"]
        .median()
        .rename("dist_month_median")
        .reset_index()
    )
    monthly = monthly.merge(dist_med, on=["Distributor_ID", "Month"], how="left")
    monthly["dist_month_rank"] = monthly.groupby(["Distributor_ID", "Month"])["monthly_volume"].rank(pct=True)
    return monthly


def build_panel(monthly: pd.DataFrame, outlet: pd.DataFrame, season: pd.DataFrame, poi_df: pd.DataFrame) -> pd.DataFrame:
    """Build (outlet, target_month) panel with features strictly before each target."""
    all_periods = sorted(monthly[["Year", "Month"]].drop_duplicates().apply(
        lambda r: int(r["Year"]) * 12 + int(r["Month"]), axis=1
    ).unique())
    all_periods = [p for p in all_periods if p != 2026 * 12 + 1]
    target_periods = all_periods[-N_TARGET_MONTHS:]

    records = []
    for p_idx in target_periods:
        yr, mo = p_idx // 12, p_idx % 12
        if mo == 0:
            yr, mo = yr - 1, 12

        train_window = monthly[monthly["Year"] * 12 + monthly["Month"] < p_idx].copy()
        if train_window.empty or train_window["Outlet_ID"].nunique() < 10:
            continue

        train_window = train_window.drop(columns=[c for c in ["dist_month_median", "dist_month_rank"] if c in train_window.columns])
        train_window = add_dist_stats(train_window)

        feats = build_outlet_features(train_window, target_month=mo)
        feats = feats.merge(outlet[["Outlet_ID", "Outlet_Type", "Outlet_Size", "Cooler_Count"]], on="Outlet_ID", how="left")

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
        feats["period_idx"] = p_idx

        if not poi_df.empty:
            feats = feats.merge(poi_df, on="Outlet_ID", how="left")

        actual = (
            monthly[(monthly["Year"] == yr) & (monthly["Month"] == mo)]
            .groupby("Outlet_ID")["monthly_volume"]
            .sum()
            .rename("actual_vol")
            .reset_index()
        )
        feats = feats.merge(actual, on="Outlet_ID", how="inner")
        feats = feats[feats["actual_vol"] > 0].copy()
        records.append(feats)

    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def enrich_panel(panel: pd.DataFrame, season: pd.DataFrame) -> pd.DataFrame:
    """Add size/type factors and peer gap (fold-safe group stats within panel slice)."""
    df = panel.copy()
    df["size_factor"] = df["Outlet_Size"].map(SIZE_POTENTIAL_FACTOR).fillna(1.0)
    df["type_factor"] = df["Outlet_Type"].map(TYPE_POTENTIAL_FACTOR).fillna(1.0)

    peer_p90 = (
        df.groupby(["Outlet_Type", "Outlet_Size", "target_month"])["hist_median_vol"]
        .quantile(0.90)
        .rename("peer_p90_t")
        .reset_index()
    )
    df = df.merge(peer_p90, on=["Outlet_Type", "Outlet_Size", "target_month"], how="left")
    df["peer_p90_t"] = df["peer_p90_t"].fillna(df["hist_median_vol"])
    df["peer_efficiency_gap"] = (df["peer_p90_t"] / (df["hist_median_vol"] + 1e-9)).clip(1.0, 3.0)
    df = df.drop(columns=["peer_p90_t"])

    df["jan_base"] = df.get("jan_hist_mean", df["hist_median_vol"])
    if "jan_hist_mean" in df.columns:
        df["jan_base"] = df["jan_hist_mean"].where(
            df["jan_hist_mean"].notna() & (df["jan_hist_mean"] > 0),
            df["hist_median_vol"],
        )
    if "competition_dampener" not in df.columns:
        df["competition_dampener"] = 1.0
    else:
        df["competition_dampener"] = df["competition_dampener"].fillna(1.0).clip(0.5, 1.0)
    return enrich_catchment_features(df)


def ceiling_target(df: pd.DataFrame) -> np.ndarray:
    """Ceiling proxy for calibration: max(p90, max, actual)."""
    p90 = df["hist_p90_vol"].values
    hmax = df["hist_max_vol"].values
    actual = df["actual_vol"].values
    return np.maximum(np.maximum(p90, hmax), actual)


def ceiling_calibration_loss(params: np.ndarray, df: pd.DataFrame) -> float:
    alpha, gamma = float(params[0]), float(params[1])
    if alpha < 0.0 or gamma < 0.0 or alpha > 2.5 or gamma > 2.0:
        return 1e9

    pred = predict_latent_potential(df, censoring_uplift=alpha, catchment_uplift=gamma)
    target = ceiling_target(df)
    cens = df["censoring_score"].values
    w = 1.0 + 2.5 * cens
    err = (np.log1p(pred) - np.log1p(target)) ** 2
    return float(np.average(err, weights=w))


def optimize_on_fold(df: pd.DataFrame) -> tuple[float, float, float]:
    res = minimize(
        ceiling_calibration_loss,
        x0=[CENSORING_UPLIFT, CATCHMENT_UPLIFT],
        args=(df,),
        method="Nelder-Mead",
        options={"maxiter": 200, "xatol": 1e-4, "fatol": 1e-5},
    )
    alpha, gamma = float(res.x[0]), float(res.x[1])
    loss = ceiling_calibration_loss(res.x, df)
    return alpha, gamma, loss


def update_latent_heuristic_file(alpha: float, gamma: float) -> None:
    path = ROOT / "pipeline" / "latent_heuristic.py"
    content = path.read_text(encoding="utf-8")
    content = re.sub(r"CENSORING_UPLIFT\s*=\s*[0-9.]+", f"CENSORING_UPLIFT = {alpha:.4f}", content)
    content = re.sub(r"CATCHMENT_UPLIFT\s*=\s*[0-9.]+", f"CATCHMENT_UPLIFT = {gamma:.4f}", content)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    logger.info("Walk-forward ceiling calibration (no LightGBM circular fit)...")

    tx = pd.read_parquet(SILVER / "transactions.parquet")
    outlet = pd.read_parquet(SILVER / "outlet_master.parquet")
    season = pd.read_parquet(SILVER / "distributor_seasonality.parquet")

    monthly = (
        tx.groupby(["Outlet_ID", "Year", "Month", "Distributor_ID"])
        .agg(monthly_volume=("Volume_Liters", "sum"))
        .reset_index()
    )

    poi_path = POI_CACHE / "poi_features.parquet"
    poi_df = pd.read_parquet(poi_path) if poi_path.exists() else pd.DataFrame()

    panel = build_panel(monthly, outlet, season, poi_df)
    if panel.empty:
        logger.error("Empty calibration panel — run silver + POI steps first.")
        sys.exit(1)

    panel = enrich_panel(panel, season)
    logger.info("Calibration panel: %s rows, %s periods", f"{len(panel):,}", panel["period_idx"].nunique())

    periods = sorted(panel["period_idx"].unique())
    tss = TimeSeriesSplit(n_splits=N_SPLITS)
    fold_records = []

    for fold_idx, (train_pos, val_pos) in enumerate(tss.split(np.arange(len(periods)))):
        train_periods = {periods[i] for i in train_pos}
        val_periods = {periods[i] for i in val_pos}
        train_df = panel[panel["period_idx"].isin(train_periods)]
        val_df = panel[panel["period_idx"].isin(val_periods)]
        if len(train_df) < 500 or len(val_df) < 200:
            continue

        alpha, gamma, train_loss = optimize_on_fold(train_df)
        val_loss = ceiling_calibration_loss([alpha, gamma], val_df)
        fold_records.append({
            "fold": fold_idx + 1,
            "CENSORING_UPLIFT": alpha,
            "CATCHMENT_UPLIFT": gamma,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_rows": len(train_df),
            "val_rows": len(val_df),
        })
        logger.info(
            "Fold %s: alpha=%.4f gamma=%.4f | train_loss=%.5f val_loss=%.5f",
            fold_idx + 1, alpha, gamma, train_loss, val_loss,
        )

    if not fold_records:
        logger.error("No calibration folds completed.")
        sys.exit(1)

    folds_df = pd.DataFrame(fold_records)
    alpha_final = float(folds_df["CENSORING_UPLIFT"].median())
    gamma_final = float(folds_df["CATCHMENT_UPLIFT"].median())

    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    calib_out = {
        "method": "walk_forward_ceiling_targets",
        "target": "max(hist_p90, hist_max, actual_vol)",
        "folds": fold_records,
        "CENSORING_UPLIFT": alpha_final,
        "CATCHMENT_UPLIFT": gamma_final,
        "previous_CENSORING_UPLIFT": CENSORING_UPLIFT,
        "previous_CATCHMENT_UPLIFT": CATCHMENT_UPLIFT,
    }
    calib_path = GOLD_DIR / "heuristic_calibration.json"
    calib_path.write_text(json.dumps(calib_out, indent=2), encoding="utf-8")
    logger.info("Saved calibration report -> %s", calib_path)

    update_latent_heuristic_file(alpha_final, gamma_final)
    logger.info("Updated latent_heuristic.py: CENSORING_UPLIFT=%.4f CATCHMENT_UPLIFT=%.4f", alpha_final, gamma_final)

    folds_df.to_csv(OUTPUT / "heuristic_calibration_folds.csv", index=False)
    logger.info("Calibration complete.")


if __name__ == "__main__":
    main()
