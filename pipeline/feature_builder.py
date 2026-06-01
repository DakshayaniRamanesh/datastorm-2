"""
DataStorm 2026 - Shared Feature Builder
======================================
Contains the canonical, high-performance vectorized feature builder to prevent DRY violations.
Includes bugfixes for out-of-bounds cross-contamination in rolling windows.
"""

import numpy as np
import pandas as pd

def build_outlet_features(monthly: pd.DataFrame, target_month: int = 1) -> pd.DataFrame:
    """Compute per-outlet historical features using vectorized operations.
    
    Includes boundary guards to prevent rolling windows from crossing outlet boundaries.
    """
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
    
    # Vectorized shift-based rolling 3-month mean with strict contiguous boundary guards
    mask3 = (df["Outlet_ID"] == df["Outlet_ID"].shift(1)) & (df["Outlet_ID"] == df["Outlet_ID"].shift(2))
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
    # Vectorized shift-based rolling 6-month mean with contiguous boundary guards
    mask6 = (
        (df["Outlet_ID"] == df["Outlet_ID"].shift(1)) & 
        (df["Outlet_ID"] == df["Outlet_ID"].shift(2)) & 
        (df["Outlet_ID"] == df["Outlet_ID"].shift(3)) & 
        (df["Outlet_ID"] == df["Outlet_ID"].shift(4)) & 
        (df["Outlet_ID"] == df["Outlet_ID"].shift(5))
    )
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

    # Rolling median & maximum features with contiguous boundary guards
    m1 = (df["Outlet_ID"] == df["Outlet_ID"].shift(1))
    m2 = (df["Outlet_ID"] == df["Outlet_ID"].shift(1)) & (df["Outlet_ID"] == df["Outlet_ID"].shift(2))
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

    # -----------------------------------------------------------------------
    # Advanced Feature Engineering Layer (Ratio, Interaction, Group Stats)
    # -----------------------------------------------------------------------
    vol_max_to_median_ratio = (hist_max_vol / (hist_median_vol + 1e-9)).clip(0.0, 100.0)
    vol_p90_to_median_ratio = (hist_p90_vol / (hist_median_vol + 1e-9)).clip(0.0, 50.0)
    vol_mean_to_median_ratio = (hist_mean_vol / (hist_median_vol + 1e-9)).clip(0.0, 10.0)
    censoring_to_cv_ratio = (censoring_score / (hist_cv + 1e-9)).clip(0.0, 50.0)

    # Load metadata safely to prevent validation loop leaks
    from pathlib import Path
    root_dir = Path(__file__).resolve().parent.parent
    outlet_file = root_dir / "pipeline" / "silver" / "outlet_master.parquet"
    if outlet_file.exists():
        outlet_df = pd.read_parquet(outlet_file).set_index("Outlet_ID")
        cooler_count = outlet_df["Cooler_Count"].reindex(hist_median_vol.index).fillna(0.0)
        outlet_type = outlet_df["Outlet_Type"].reindex(hist_median_vol.index).fillna("Grocery")
        outlet_size = outlet_df["Outlet_Size"].reindex(hist_median_vol.index).fillna("Medium")
    else:
        cooler_count = pd.Series(0.0, index=hist_median_vol.index)
        outlet_type = pd.Series("Grocery", index=hist_median_vol.index)
        outlet_size = pd.Series("Medium", index=hist_median_vol.index)

    cooler_per_volume = (cooler_count / (hist_median_vol + 1e-9)).clip(0.0, 10.0)

    # Group statistics
    group_df = pd.DataFrame({
        "Outlet_Type": outlet_type,
        "Outlet_Size": outlet_size,
        "hist_median_vol": hist_median_vol,
        "hist_max_vol": hist_max_vol,
        "hist_cv": hist_cv
    })
    gp_stats = group_df.groupby(["Outlet_Type", "Outlet_Size"])
    group_mean_vol = gp_stats["hist_median_vol"].transform("mean").reindex(hist_median_vol.index).fillna(0.0)
    group_max_vol = gp_stats["hist_max_vol"].transform("mean").reindex(hist_median_vol.index).fillna(0.0)
    group_cv = gp_stats["hist_cv"].transform("mean").reindex(hist_median_vol.index).fillna(0.0)

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
        "rolling_maximum": rolling_maximum,
        # Advanced Features
        "vol_max_to_median_ratio": vol_max_to_median_ratio,
        "vol_p90_to_median_ratio": vol_p90_to_median_ratio,
        "vol_mean_to_median_ratio": vol_mean_to_median_ratio,
        "censoring_to_cv_ratio": censoring_to_cv_ratio,
        "cooler_per_volume": cooler_per_volume,
        "group_mean_vol": group_mean_vol,
        "group_max_vol": group_max_vol,
        "group_cv": group_cv
    }).reset_index()

