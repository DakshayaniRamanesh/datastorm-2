import time
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
SILVER = ROOT / "pipeline" / "silver"

# Original grouping logic
def original_features(grp):
    vols = grp["monthly_volume"].values
    n = len(vols)
    months = grp["Month"].values

    mean_vol   = float(np.mean(vols))
    median_vol = float(np.median(vols))
    max_vol    = float(np.max(vols))
    p75_vol    = float(np.percentile(vols, 75))
    p90_vol    = float(np.percentile(vols, 90))
    std_vol    = float(np.std(vols))
    cv         = std_vol / (mean_vol + 1e-9)

    plateau_score = 0.0
    if n >= 3:
        cvs = []
        for i in range(n - 2):
            w = vols[i:i+3]
            cvs.append(np.std(w) / (np.mean(w) + 1e-9))
        plateau_score = float(np.mean(np.array(cvs) < 0.10))

    dist_med = grp["dist_month_median"].values
    at_cap   = float(np.mean(np.abs(vols - dist_med) / (dist_med + 1e-9) < 0.15))

    yearly     = grp.groupby("Year")["monthly_volume"].mean()
    stagnation = 0.0
    yoy_growth = 0.0
    if len(yearly) >= 2:
        rates      = yearly.pct_change().dropna().values
        stagnation = float(np.mean(np.abs(rates) < 0.05))
        yoy_growth = float(np.clip(rates.mean(), -0.30, 0.50))

    cv_score = float(max(0.0, (0.30 - cv) / 0.30))

    q4_vols = vols[np.isin(months, [10, 11, 12])]
    if len(q4_vols) >= 2:
        q4_prem = q4_vols.mean() / (mean_vol + 1e-9) - 1.0
        q4_supp = float(np.clip(-q4_prem * 2.0, 0, 1))
    else:
        q4_supp = 0.0

    censoring_score = (
        0.30 * plateau_score
        + 0.25 * at_cap
        + 0.20 * stagnation
        + 0.15 * cv_score
        + 0.10 * q4_supp
    )

    if n >= 3 and max_vol > 0:
        rolling_3m_means = [np.mean(vols[i:i+3]) for i in range(n - 2)]
        capacity_proximity_ratio = float(np.mean(np.array(rolling_3m_means) / (max_vol + 1e-9)))
    else:
        capacity_proximity_ratio = float(mean_vol / (max_vol + 1e-9))
    capacity_proximity_ratio = float(np.clip(capacity_proximity_ratio, 0.0, 1.0))

    if n >= 6:
        rolling_6m_stds = [np.std(vols[i:i+6]) / (np.mean(vols[i:i+6]) + 1e-9) for i in range(n - 5)]
        purchase_pace_variance = float(np.mean(rolling_6m_stds))
    else:
        purchase_pace_variance = cv

    dist_rank_mean = float(grp["dist_month_rank"].mean())

    jan_vols      = vols[np.isin(months, [1])]
    jan_hist_mean = float(jan_vols.mean()) if len(jan_vols) > 0 else mean_vol

    return pd.Series({
        "hist_mean_vol":            mean_vol,
        "hist_median_vol":          median_vol,
        "hist_max_vol":             max_vol,
        "hist_p75_vol":             p75_vol,
        "hist_p90_vol":             p90_vol,
        "hist_std_vol":             std_vol,
        "hist_cv":                  cv,
        "hist_months":              n,
        "censoring_score":          float(np.clip(censoring_score, 0, 1)),
        "cens_plateau":             plateau_score,
        "cens_dist_cap":            at_cap,
        "cens_stagnation":          stagnation,
        "cens_cv_score":            cv_score,
        "cens_q4_suppression":      q4_supp,
        "yoy_growth":               yoy_growth,
        "jan_hist_mean":            jan_hist_mean,
        "capacity_proximity_ratio": capacity_proximity_ratio,
        "purchase_pace_variance":   purchase_pace_variance,
        "dist_rank_mean":           dist_rank_mean,
        "primary_dist":             grp.groupby("Distributor_ID")["monthly_volume"].sum().idxmax(),
    })

# Vectorized logic
def vectorized_features(monthly, target_month=1):
    df = monthly.copy()
    df = df.sort_values(["Outlet_ID", "Year", "Month"])
    
    # 1. Base rollups
    gp = df.groupby("Outlet_ID")
    
    hist_mean_vol = gp["monthly_volume"].mean()
    hist_median_vol = gp["monthly_volume"].median()
    hist_max_vol = gp["monthly_volume"].max()
    hist_p75_vol = gp["monthly_volume"].quantile(0.75)
    hist_p90_vol = gp["monthly_volume"].quantile(0.90)
    
    # Compute std with ddof=0 (population standard deviation to match numpy)
    hist_std_vol = gp["monthly_volume"].std(ddof=0).fillna(0)
    hist_months = gp["monthly_volume"].count()
    hist_cv = hist_std_vol / (hist_mean_vol + 1e-9)
    
    # 2. Plateau Flag (rolling 3-month CV < 0.10, ddof=0)
    # Using the algebraic formula for population variance: Var = E[X^2] - E[X]^2
    roll_mean = df.groupby("Outlet_ID")["monthly_volume"].transform(lambda x: x.rolling(3, min_periods=3).mean())
    roll_mean_sq = df.groupby("Outlet_ID")["monthly_volume"].transform(lambda x: x.pow(2).rolling(3, min_periods=3).mean())
    roll_var = (roll_mean_sq - roll_mean.pow(2)).clip(lower=0)
    roll_std = np.sqrt(roll_var)
    roll_cv = roll_std / (roll_mean + 1e-9)
    df["plateau_flag"] = (roll_cv < 0.10).astype(float)
    
    # Set plateau_flag to NaN for groups with count < 3
    df.loc[df["Outlet_ID"].isin(hist_months[hist_months < 3].index), "plateau_flag"] = np.nan
    cens_plateau = df.groupby("Outlet_ID")["plateau_flag"].mean().fillna(0.0)
    
    # 3. Dist cap flag (tracking distributor delivery cap)
    df["at_cap_flag"] = (np.abs(df["monthly_volume"] - df["dist_month_median"]) / (df["dist_month_median"] + 1e-9) < 0.15).astype(float)
    cens_dist_cap = df.groupby("Outlet_ID")["at_cap_flag"].mean().fillna(0.0)
    
    # 4. Inter-year stagnation & growth
    # Group by Outlet_ID and Year, take mean volume
    yearly = df.groupby(["Outlet_ID", "Year"])["monthly_volume"].mean().reset_index()
    yearly = yearly.sort_values(["Outlet_ID", "Year"])
    yearly["growth"] = yearly.groupby("Outlet_ID")["monthly_volume"].pct_change()
    
    # Set stagnation_flag to NaN where growth is NaN, otherwise check if growth < 0.05
    yearly["stagnation_flag"] = np.where(yearly["growth"].isna(), np.nan, (np.abs(yearly["growth"]) < 0.05).astype(float))
    
    # Stagnation is computed only over valid growth rates (excluding the first year which is NaN)
    # Groupby mean ignores NaN automatically
    cens_stagnation = yearly.groupby("Outlet_ID")["stagnation_flag"].mean().fillna(0.0)
    
    # Wait, in the original code:
    # rates = yearly.pct_change().dropna().values
    # stagnation = float(np.mean(np.abs(rates) < 0.05))
    # Let's verify if there is any difference due to groupby order or missing years
    # If an outlet has missing years, groupby might have fewer rows, which is handled correctly
    yoy_growth = yearly.groupby("Outlet_ID")["growth"].mean().fillna(0.0).clip(-0.30, 0.50)
    
    # 5. Low overall CV
    cens_cv_score = ((0.30 - hist_cv) / 0.30).clip(lower=0.0)
    
    # 6. Q4 suppression
    df["q4_volume"] = np.where(df["Month"].isin([10, 11, 12]), df["monthly_volume"], np.nan)
    q4_mean = df.groupby("Outlet_ID")["q4_volume"].mean()
    q4_count = df.groupby("Outlet_ID")["q4_volume"].count()
    q4_prem = q4_mean / (hist_mean_vol + 1e-9) - 1.0
    cens_q4_suppression = (-q4_prem * 2.0).clip(0.0, 1.0).fillna(0.0)
    # Set to 0.0 if there are less than 2 Q4 months available
    cens_q4_suppression = np.where(q4_count >= 2, cens_q4_suppression, 0.0)
    
    # 7. Composite censoring score
    censoring_score = (
        0.30 * cens_plateau
        + 0.25 * cens_dist_cap
        + 0.20 * cens_stagnation
        + 0.15 * cens_cv_score
        + 0.10 * cens_q4_suppression
    ).clip(0.0, 1.0)
    
    # 8. Capacity Proximity Ratio
    roll_mean_avg = roll_mean.groupby(df["Outlet_ID"]).mean()
    capacity_proximity_ratio = roll_mean_avg / (hist_max_vol + 1e-9)
    capacity_proximity_ratio = np.where(hist_months >= 3, capacity_proximity_ratio, hist_mean_vol / (hist_max_vol + 1e-9))
    capacity_proximity_ratio = np.clip(capacity_proximity_ratio, 0.0, 1.0)
    
    # 9. Purchase Pace Variance (rolling 6-month CV average, ddof=0)
    roll6_mean = df.groupby("Outlet_ID")["monthly_volume"].transform(lambda x: x.rolling(6, min_periods=6).mean())
    roll6_mean_sq = df.groupby("Outlet_ID")["monthly_volume"].transform(lambda x: x.pow(2).rolling(6, min_periods=6).mean())
    roll6_var = (roll6_mean_sq - roll6_mean.pow(2)).clip(lower=0)
    roll6_std = np.sqrt(roll6_var)
    roll6_cv = roll6_std / (roll6_mean + 1e-9)
    purchase_pace_variance = roll6_cv.groupby(df["Outlet_ID"]).mean()
    purchase_pace_variance = np.where(hist_months >= 6, purchase_pace_variance, hist_cv)
    
    # 10. Distributor Rank mean
    dist_rank_mean = df.groupby("Outlet_ID")["dist_month_rank"].mean()
    
    # 11. Target month historical mean
    target_vols = df[df["Month"] == target_month]
    target_mean = target_vols.groupby("Outlet_ID")["monthly_volume"].mean()
    # Align and fill with overall mean
    target_mean = target_mean.reindex(hist_mean_vol.index).fillna(hist_mean_vol)
    
    # 12. Primary distributor
    dist_sums = df.groupby(["Outlet_ID", "Distributor_ID"])["monthly_volume"].sum().reset_index()
    primary_dist = dist_sums.sort_values("monthly_volume").groupby("Outlet_ID")["Distributor_ID"].last()
    
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

if __name__ == "__main__":
    print("Loading datasets...")
    tx = pd.read_parquet(SILVER / "transactions.parquet")
    monthly = (
        tx.groupby(["Outlet_ID", "Year", "Month", "Distributor_ID"])
        .agg(
            monthly_volume=("Volume_Liters", "sum"),
            monthly_revenue=("Total_Bill_Value", "sum"),
            sku_count=("SKU_ID", "nunique"),
            transaction_count=("SKU_ID", "count"),
        )
        .reset_index()
    )
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

    # Sample for verification
    sample_outlets = monthly["Outlet_ID"].unique()[:500]
    monthly_sample = monthly[monthly["Outlet_ID"].isin(sample_outlets)].copy()
    
    print(f"Comparing on {len(sample_outlets)} outlets...")
    
    t0 = time.time()
    res_orig = monthly_sample.groupby("Outlet_ID").apply(original_features).reset_index()
    t_orig = time.time() - t0
    print(f"Original group-by took: {t_orig:.4f}s")
    
    t0 = time.time()
    res_vect = vectorized_features(monthly_sample)
    t_vect = time.time() - t0
    print(f"Vectorized features took: {t_vect:.4f}s")
    print(f"Speedup: {t_orig / t_vect:.1f}x")
    
    # Merge for direct comparison
    comp = res_orig.merge(res_vect, on="Outlet_ID", suffixes=("_orig", "_vect"))
    
    print("\nAbsolute differences:")
    for col in res_orig.columns:
        if col == "Outlet_ID":
            continue
        if col == "primary_dist":
            # check equality
            n_diff = (comp[col + "_orig"] != comp[col + "_vect"]).sum()
            print(f"  {col:<25} errors: {n_diff}")
        else:
            diff = np.abs(comp[col + "_orig"] - comp[col + "_vect"])
            print(f"  {col:<25} max_diff: {diff.max():.6e} | mean_diff: {diff.mean():.6e}")
            
    # Print sample discrepancies for cens_stagnation or cens_plateau
    discrepant = comp[np.abs(comp["cens_stagnation_orig"] - comp["cens_stagnation_vect"]) > 1e-5]
    if not discrepant.empty:
        print("\nDiscrepancy Sample for cens_stagnation:")
        for idx, row in discrepant.head(3).iterrows():
            oid = row["Outlet_ID"]
            print(f"\nOutlet: {oid}")
            print(f"  Original stagnation:   {row['cens_stagnation_orig']}")
            print(f"  Vectorized stagnation: {row['cens_stagnation_vect']}")
            
            # Print actual volumes
            grp = monthly_sample[monthly_sample["Outlet_ID"] == oid].sort_values(["Year", "Month"])
            print("  Raw records:")
            print(grp[["Year", "Month", "monthly_volume"]].to_string(index=False))
            
            # Original calculation details
            yearly_orig = grp.groupby("Year")["monthly_volume"].mean()
            rates_orig = yearly_orig.pct_change().dropna().values
            print(f"  Original yearly means:\n{yearly_orig.to_dict()}")
            print(f"  Original rates: {rates_orig}")
            
            # Vectorized calculation details
            yearly_vect = grp.groupby("Year")["monthly_volume"].mean().reset_index()
            yearly_vect["growth"] = yearly_vect["monthly_volume"].pct_change()
            yearly_vect["stagnation_flag"] = (np.abs(yearly_vect["growth"]) < 0.05).astype(float)
            print(f"  Vectorized growth:\n{yearly_vect[['Year', 'monthly_volume', 'growth', 'stagnation_flag']].to_string(index=False)}")
