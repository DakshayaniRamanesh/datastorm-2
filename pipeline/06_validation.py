"""
OUT-OF-TIME VALIDATION - LightGBM Chronological Holdout Framework
==================================================================
Implements a strict out-of-time validation protocol for the LightGBM
hybrid demand forecasting model to eliminate chronological data leakage.

APPROACH: TimeSeriesSplit + Out-of-Time Anchor Holdout
-------------------------------------------------------
1. Aggregate transactions to monthly outlet level.
2. Use sklearn TimeSeriesSplit (n_splits=6) to create sequential folds.
3. For each fold:
   a. Build features using only the training window (no leakage).
   b. Train a LightGBM regressor on (features → log1p(volume)) pairs
      within the training window (using multiple target months).
   c. Build prediction features at the training window boundary and
      predict volumes for the validation months.
   d. Compare predictions to actual observed volumes.
4. Final holdout: train up to Oct 2025, validate on Nov-Dec 2025.
5. Report MAE, RMSE, MedAE and MAPE per fold and overall.

Metrics are saved to output/validation_report.csv.

Usage:
    python pipeline/06_validation.py
"""

import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb

warnings.filterwarnings("ignore")

ROOT     = Path(__file__).parent.parent
SILVER   = ROOT / "pipeline" / "silver"
GOLD_DIR = ROOT / "pipeline" / "gold"
OUTPUT   = ROOT / "output"
OUTPUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_SPLITS          = 6      # TimeSeriesSplit folds
MIN_TRAIN_MONTHS  = 6      # minimum months required in a training window
N_TRAIN_TARGETS   = 6      # how many trailing months to use as training labels per fold
HOLDOUT_MONTHS    = [      # final out-of-time holdout (proxy for Jan 2026)
    (2025, 11), (2025, 12)
]

SEASONALITY_MULTIPLIER = {
    "Favorable": 1.15, "Moderate": 1.00, "Un-Favorable": 0.88,
}

LGB_PARAMS = {
    "objective":        "regression",
    "metric":           "rmse",
    "n_estimators":     400,
    "learning_rate":    0.05,
    "num_leaves":       31,
    "min_child_samples":15,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":        0.1,
    "reg_lambda":       0.2,
    "verbose":          -1,
    "n_jobs":           -1,
    "random_state":     42,
}


# ---------------------------------------------------------------------------
# Vectorised Feature Builder (mirrors 04_gold_features_model.py exactly)
# ---------------------------------------------------------------------------

def build_outlet_features(monthly: pd.DataFrame, target_month: int = 1) -> pd.DataFrame:
    """
    Compute per-outlet historical features using fully vectorized pandas operations.
    Mirrors the feature builder in 04_gold_features_model.py.
    """
    df = monthly.copy().sort_values(["Outlet_ID", "Year", "Month"])
    gp = df.groupby("Outlet_ID")

    hist_mean_vol   = gp["monthly_volume"].mean()
    hist_median_vol = gp["monthly_volume"].median()
    hist_max_vol    = gp["monthly_volume"].max()
    hist_p75_vol    = gp["monthly_volume"].quantile(0.75)
    hist_p90_vol    = gp["monthly_volume"].quantile(0.90)
    hist_std_vol    = gp["monthly_volume"].std(ddof=0).fillna(0)
    hist_months     = gp["monthly_volume"].count()
    hist_cv         = hist_std_vol / (hist_mean_vol + 1e-9)

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
    df.loc[df["Outlet_ID"].isin(hist_months[hist_months < 3].index), "plateau_flag"] = np.nan
    cens_plateau  = df.groupby("Outlet_ID")["plateau_flag"].mean().fillna(0.0)

    df["at_cap_flag"] = (
        np.abs(df["monthly_volume"] - df["dist_month_median"])
        / (df["dist_month_median"] + 1e-9) < 0.15
    ).astype(float)
    cens_dist_cap = df.groupby("Outlet_ID")["at_cap_flag"].mean().fillna(0.0)

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

    cens_cv_score = ((0.30 - hist_cv) / 0.30).clip(lower=0.0)

    df["q4_vol"] = np.where(df["Month"].isin([10, 11, 12]), df["monthly_volume"], np.nan)
    q4_mean  = df.groupby("Outlet_ID")["q4_vol"].mean()
    q4_count = df.groupby("Outlet_ID")["q4_vol"].count()
    q4_prem  = q4_mean / (hist_mean_vol + 1e-9) - 1.0
    cens_q4_raw = (-q4_prem * 2.0).clip(0.0, 1.0).fillna(0.0)
    cens_q4_suppression = pd.Series(
        np.where(q4_count >= 2, cens_q4_raw, 0.0),
        index=hist_mean_vol.index
    )

    censoring_score = (
        0.30 * cens_plateau
        + 0.25 * cens_dist_cap
        + 0.20 * cens_stagnation
        + 0.15 * cens_cv_score
        + 0.10 * cens_q4_suppression
    ).clip(0.0, 1.0)

    roll3_mean_avg           = roll3_mean.groupby(df["Outlet_ID"]).mean()
    capacity_proximity_ratio = pd.Series(
        np.clip(
            np.where(
                hist_months >= 3,
                roll3_mean_avg / (hist_max_vol + 1e-9),
                hist_mean_vol  / (hist_max_vol + 1e-9)
            ), 0.0, 1.0
        ), index=hist_mean_vol.index
    )

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

    dist_rank_mean = df.groupby("Outlet_ID")["dist_month_rank"].mean()

    target_vols = df[df["Month"] == target_month]
    target_mean = (
        target_vols.groupby("Outlet_ID")["monthly_volume"].mean()
        .reindex(hist_mean_vol.index).fillna(hist_mean_vol)
    )

    dist_sums    = (
        df.groupby(["Outlet_ID", "Distributor_ID"])["monthly_volume"]
        .sum().reset_index()
    )
    primary_dist = (
        dist_sums.sort_values("monthly_volume")
        .groupby("Outlet_ID")["Distributor_ID"].last()
    )

    return pd.DataFrame({
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


FEATURE_COLS = [
    "hist_mean_vol", "hist_median_vol", "hist_max_vol",
    "hist_p75_vol", "hist_p90_vol", "hist_std_vol",
    "hist_cv", "hist_months",
    "censoring_score", "cens_plateau", "cens_dist_cap",
    "cens_stagnation", "cens_cv_score", "cens_q4_suppression",
    "capacity_proximity_ratio", "purchase_pace_variance", "dist_rank_mean",
    "yoy_growth", "target_season_factor", "target_month",
]


def add_dist_stats(monthly: pd.DataFrame) -> pd.DataFrame:
    """Compute dist_month_median and dist_month_rank within a training window."""
    dist_med = (
        monthly.groupby(["Distributor_ID", "Month"])["monthly_volume"]
        .median().rename("dist_month_median").reset_index()
    )
    monthly = monthly.merge(dist_med, on=["Distributor_ID", "Month"], how="left")
    monthly["dist_month_rank"] = (
        monthly.groupby(["Distributor_ID", "Month"])["monthly_volume"]
        .rank(pct=True)
    )
    return monthly


def build_training_panel(
    monthly: pd.DataFrame,
    outlet: pd.DataFrame,
    season: pd.DataFrame,
    train_periods: set,
    n_train_targets: int = N_TRAIN_TARGETS,
) -> pd.DataFrame:
    """
    Build labelled training records from within `train_periods`.

    For the last `n_train_targets` months in the training window:
      - Features are built from all months strictly before the target month.
      - Target is log1p(observed volume in that month).
    """
    sorted_periods = sorted(train_periods)
    target_periods = sorted_periods[-n_train_targets:]

    records = []
    for p in target_periods:
        yr, mo = p // 12, p % 12
        if mo == 0:
            yr, mo = yr - 1, 12

        cutoff = p
        train_window = monthly[monthly["period_idx"] < cutoff].copy()
        if train_window.empty or train_window["Outlet_ID"].nunique() < 5:
            continue

        train_window = add_dist_stats(train_window)
        feats = build_outlet_features(train_window, target_month=mo)
        feats = feats.merge(
            outlet[["Outlet_ID", "Outlet_Type", "Outlet_Size"]],
            on="Outlet_ID", how="left"
        )

        # Seasonality
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
        feats["target_seasonality"]   = feats.get("target_seasonality", "Moderate").fillna("Moderate")
        feats["target_season_factor"] = feats["target_seasonality"].map(
            SEASONALITY_MULTIPLIER
        ).fillna(1.0)
        feats["target_month"] = mo

        # Actual observed volumes in this target month
        actual = (
            monthly[monthly["period_idx"] == p]
            .groupby("Outlet_ID")["monthly_volume"].sum()
            .reset_index()
            .rename(columns={"monthly_volume": "actual_vol"})
        )
        feats = feats.merge(actual, on="Outlet_ID", how="inner")
        feats = feats[feats["actual_vol"] > 0].copy()
        feats["target_log_vol"] = np.log1p(feats["actual_vol"])
        records.append(feats)

    if not records:
        return pd.DataFrame()
    return pd.concat(records, ignore_index=True)


def train_lgb_model(train_df: pd.DataFrame) -> lgb.LGBMRegressor:
    """Train LightGBM on prepared training panel."""
    feat_cols_avail = [c for c in FEATURE_COLS if c in train_df.columns]
    X = train_df[feat_cols_avail].fillna(0)
    y = train_df["target_log_vol"].values
    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(X, y)
    return model, feat_cols_avail


def predict_for_periods(
    model: lgb.LGBMRegressor,
    feat_cols: list,
    monthly: pd.DataFrame,
    outlet: pd.DataFrame,
    season: pd.DataFrame,
    train_periods: set,
    val_periods: set,
) -> pd.DataFrame:
    """
    For each period in val_periods, build features from train_periods and predict.
    Returns a DataFrame with Outlet_ID, period, prediction, censoring_score.
    """
    results = []
    # Use the full training data for feature computation
    train_monthly = monthly[monthly["period_idx"].isin(train_periods)].copy()
    train_monthly = add_dist_stats(train_monthly)

    for p in sorted(val_periods):
        yr, mo = p // 12, p % 12
        if mo == 0:
            yr, mo = yr - 1, 12

        feats = build_outlet_features(train_monthly, target_month=mo)
        feats = feats.merge(
            outlet[["Outlet_ID", "Outlet_Type", "Outlet_Size"]],
            on="Outlet_ID", how="left"
        )
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
        feats["target_seasonality"]   = feats.get("target_seasonality", "Moderate").fillna("Moderate")
        feats["target_season_factor"] = feats["target_seasonality"].map(
            SEASONALITY_MULTIPLIER
        ).fillna(1.0)
        feats["target_month"] = mo

        feat_cols_avail = [c for c in feat_cols if c in feats.columns]
        X = feats[feat_cols_avail].fillna(0)
        log_pred = model.predict(X)
        feats["prediction"]  = np.clip(np.expm1(log_pred), 1.0, None)
        feats["period_idx"]  = p
        results.append(feats[["Outlet_ID", "period_idx", "prediction", "censoring_score"]])

    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True)


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Compute MAE, RMSE, MedAE, and MAPE."""
    errors  = predicted - actual
    abs_err = np.abs(errors)
    return {
        "MAE":    float(np.mean(abs_err)),
        "RMSE":   float(np.sqrt(np.mean(errors ** 2))),
        "MedAE":  float(np.median(abs_err)),
        "MAPE_%": float(np.mean(abs_err / (np.abs(actual) + 1e-9)) * 100),
    }


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("OUT-OF-TIME VALIDATION - LightGBM Chronological Holdout")
    print("=" * 60)
    print(f"  TimeSeriesSplit folds:   {N_SPLITS}")
    print(f"  Training targets/fold:   {N_TRAIN_TARGETS} months")
    print(f"  Final holdout months:    {HOLDOUT_MONTHS}")

    # Load data
    print("\n[1/4] Loading Silver data...")
    tx     = pd.read_parquet(SILVER / "transactions.parquet")
    outlet = pd.read_parquet(SILVER / "outlet_master.parquet")
    season = pd.read_parquet(SILVER / "distributor_seasonality.parquet")

    # Monthly aggregation
    print("[2/4] Aggregating to monthly outlet level...")
    monthly = (
        tx.groupby(["Outlet_ID", "Year", "Month", "Distributor_ID"])
        .agg(monthly_volume=("Volume_Liters", "sum"))
        .reset_index()
    )
    monthly["period_idx"] = monthly["Year"] * 12 + monthly["Month"]
    all_periods           = sorted(monthly["period_idx"].unique())
    n_periods             = len(all_periods)
    print(f"  Total periods: {n_periods}  "
          f"(from {min(all_periods)//12}-{min(all_periods)%12:02d} "
          f"to {max(all_periods)//12}-{max(all_periods)%12:02d})")

    period_to_pos = {p: i for i, p in enumerate(all_periods)}
    monthly["period_pos"] = monthly["period_idx"].map(period_to_pos)

    # -----------------------------------------------------------------------
    # TimeSeriesSplit rolling-window validation
    # -----------------------------------------------------------------------
    print(f"\n[3/4] Running {N_SPLITS}-fold out-of-time validation with LightGBM...")
    tss              = TimeSeriesSplit(n_splits=N_SPLITS, gap=0)
    period_positions = np.arange(n_periods)
    fold_results     = []

    for fold_idx, (train_pos_idx, val_pos_idx) in enumerate(tss.split(period_positions)):
        if len(train_pos_idx) < MIN_TRAIN_MONTHS:
            continue

        train_periods = set(all_periods[i] for i in train_pos_idx)
        val_periods   = set(all_periods[i] for i in val_pos_idx)

        t_min = min(train_periods)
        t_max = max(train_periods)
        v_min = min(val_periods)
        v_max = max(val_periods)
        label = (f"  Fold {fold_idx+1}: train "
                 f"{t_min//12}-{t_min%12:02d} -> {t_max//12}-{t_max%12:02d} | "
                 f"validate {v_min//12}-{v_min%12:02d} -> {v_max//12}-{v_max%12:02d}")
        print(label)

        # Build training panel
        train_df = build_training_panel(monthly, outlet, season, train_periods, N_TRAIN_TARGETS)
        if train_df.empty:
            print("    [SKIP] Insufficient training data.")
            continue

        # Train LightGBM
        model, feat_cols = train_lgb_model(train_df)

        # Predict on validation periods
        preds_df = predict_for_periods(
            model, feat_cols, monthly, outlet, season, train_periods, val_periods
        )
        if preds_df.empty:
            continue

        # Actuals: mean observed volume across validation periods per outlet
        actuals = (
            monthly[monthly["period_idx"].isin(val_periods)]
            .groupby("Outlet_ID")["monthly_volume"]
            .mean()
            .reset_index()
            .rename(columns={"monthly_volume": "actual"})
        )
        # Average predictions across validation periods per outlet
        preds_agg = (
            preds_df.groupby("Outlet_ID")["prediction"]
            .mean()
            .reset_index()
        )
        merged = preds_agg.merge(actuals, on="Outlet_ID", how="inner")
        if merged.empty:
            continue

        metrics = compute_metrics(merged["actual"].values, merged["prediction"].values)
        metrics.update({
            "fold":           fold_idx + 1,
            "train_periods":  len(train_periods),
            "val_periods":    len(val_periods),
            "outlets_scored": len(merged),
            "train_end":      f"{t_max//12}-{t_max%12:02d}",
            "val_window":     f"{v_min//12}-{v_min%12:02d} -> {v_max//12}-{v_max%12:02d}",
        })
        fold_results.append(metrics)
        print(f"    MAE={metrics['MAE']:,.1f} L  |  RMSE={metrics['RMSE']:,.1f} L  |  "
              f"MedAE={metrics['MedAE']:,.1f} L  |  MAPE={metrics['MAPE_%']:.1f}%  |  "
              f"n={metrics['outlets_scored']:,}")

    # -----------------------------------------------------------------------
    # Final out-of-time holdout (Nov-Dec 2025 as Jan 2026 proxy)
    # -----------------------------------------------------------------------
    print(f"\n[4/4] Final holdout: train up to Oct 2025, validate Nov-Dec 2025...")
    holdout_period_idxs = set(y * 12 + m for y, m in HOLDOUT_MONTHS)
    cutoff_period       = min(holdout_period_idxs) - 1   # Oct 2025

    final_train_periods = set(p for p in all_periods if p <= cutoff_period)
    final_val_periods   = holdout_period_idxs

    final_train_df = build_training_panel(
        monthly, outlet, season, final_train_periods, N_TRAIN_TARGETS
    )
    final_metrics = {}
    if not final_train_df.empty:
        final_model, final_feat_cols = train_lgb_model(final_train_df)
        final_preds_df = predict_for_periods(
            final_model, final_feat_cols,
            monthly, outlet, season,
            final_train_periods, final_val_periods
        )
        actuals_final = (
            monthly[monthly["period_idx"].isin(final_val_periods)]
            .groupby("Outlet_ID")["monthly_volume"]
            .mean()
            .reset_index()
            .rename(columns={"monthly_volume": "actual"})
        )
        preds_final_agg = (
            final_preds_df.groupby("Outlet_ID")["prediction"]
            .mean()
            .reset_index()
        )
        merged_final = preds_final_agg.merge(actuals_final, on="Outlet_ID", how="inner")

        if not merged_final.empty:
            final_metrics = compute_metrics(
                merged_final["actual"].values,
                merged_final["prediction"].values
            )
            final_metrics.update({
                "fold":           "FINAL_HOLDOUT",
                "train_periods":  len(final_train_periods),
                "val_periods":    len(final_val_periods),
                "outlets_scored": len(merged_final),
                "train_end":      "2025-10",
                "val_window":     "2025-11 -> 2025-12",
            })
            fold_results.append(final_metrics)
            print(f"  FINAL HOLDOUT:  MAE={final_metrics['MAE']:,.1f} L  |  "
                  f"RMSE={final_metrics['RMSE']:,.1f} L  |  "
                  f"MedAE={final_metrics['MedAE']:,.1f} L  |  "
                  f"MAPE={final_metrics['MAPE_%']:.1f}%")

    # -----------------------------------------------------------------------
    # Save and summarize
    # -----------------------------------------------------------------------
    results_df = pd.DataFrame(fold_results)
    out_path   = OUTPUT / "validation_report.csv"
    results_df.to_csv(out_path, index=False)

    print(f"\n[SUMMARY] Cross-fold averages (excluding final holdout):")
    cv_rows = results_df[results_df["fold"] != "FINAL_HOLDOUT"]
    if not cv_rows.empty:
        print(f"  Avg MAE:   {cv_rows['MAE'].mean():,.1f} L")
        print(f"  Avg RMSE:  {cv_rows['RMSE'].mean():,.1f} L")
        print(f"  Avg MedAE: {cv_rows['MedAE'].mean():,.1f} L")
        print(f"  Avg MAPE:  {cv_rows['MAPE_%'].mean():.1f}%")

    print(f"\n  Validation report saved -> {out_path}")
    print("\n[OK]  Out-of-time validation complete.\n")


if __name__ == "__main__":
    main()
