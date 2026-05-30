"""
DataStorm 2026 - Budget Optimization Pipeline
=============================================
Filters outlets for the Western Province, calculates their Latent Potential Gap,
and uses a dual bisection method (or SciPy SLSQP fallback) to allocate a
LKR 5,000,000 budget to maximize the overall incremental volume lift.

Objective:
  Maximize Sum( Potential_Gap_i * Alpha * ln(1 + Spend_i / Beta) )
  where Alpha = 0.15, Beta = 10000.

Constraints:
  1. Sum( Spend_i ) <= 5,000,000 LKR
  2. 0 <= Spend_i <= 100,000 LKR (per-outlet cap for distribution parity)
"""

import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("BudgetOptimizer")

# Paths
ROOT = Path(__file__).parent.parent
GOLD = ROOT / "pipeline" / "gold"
OUTPUT = ROOT / "output"
OUTPUT.mkdir(parents=True, exist_ok=True)

# Parameters
BUDGET_LKR = 5000000.0
MAX_SPEND_PER_OUTLET_LKR = 100000.0  # Parity cap per outlet
ALPHA = 0.15
BETA = 10000.0

def allocate_budget_bisection(
    potentials: np.ndarray,
    total_budget: float,
    alpha: float,
    beta: float,
    max_spend: float
) -> np.ndarray:
    """Solve the bounded convex resource allocation problem using dual bisection.

    Extremely fast, robust, and handles boundaries exactly.
    """
    N = len(potentials)
    if N == 0 or np.sum(potentials) <= 0:
        return np.zeros(N)

    # Search bounds for the dual variable lambda (marginal utility)
    low_lambda = 0.0
    high_lambda = float(np.max(potentials) * alpha / beta) + 1e-9

    for _ in range(100):
        mid_lambda = (low_lambda + high_lambda) / 2.0
        
        # S_i = P_i * alpha / lambda - beta
        spends = (potentials * alpha / (mid_lambda + 1e-15)) - beta
        spends = np.clip(spends, 0.0, max_spend)
        
        current_sum = np.sum(spends)
        if current_sum > total_budget:
            low_lambda = mid_lambda  # Need to reduce spend (increase lambda)
        else:
            high_lambda = mid_lambda  # Need to increase spend (decrease lambda)

    final_spends = (potentials * alpha / (low_lambda + 1e-15)) - beta
    return np.clip(final_spends, 0.0, max_spend)

def main():
    logger.info("Initializing Budget Allocation Optimizer...")
    
    # 1. Load Gold features
    gold_path = GOLD / "gold_features.parquet"
    if not gold_path.exists():
        logger.error(f"Gold feature table not found at: {gold_path}. Run model pipeline first.")
        sys.exit(1)
        
    df = pd.read_parquet(gold_path)
    logger.info(f"Loaded Gold features: {len(df):,} total outlets.")
    
    # 2. Filter for Western Province (Primary Distributor in DIST_W_01, DIST_W_02, DIST_W_03)
    wp_mask = df["primary_dist"].astype(str).str.startswith("DIST_W_")
    wp_df = df[wp_mask].copy()
    logger.info(f"Filtered for Western Province: {len(wp_df):,} outlets.")
    
    if len(wp_df) == 0:
        logger.error("No outlets found in Western Province. Allocation aborted.")
        sys.exit(1)
        
    # 3. Calculate Potential Gap (Potential - Historical sales)
    # Potential Gap = Maximum_Monthly_Liters - hist_median_vol
    wp_df["potential_gap"] = (wp_df["Maximum_Monthly_Liters"] - wp_df["hist_median_vol"]).clip(lower=0.0)
    
    potentials = wp_df["potential_gap"].values
    
    # 4. Run Optimization
    logger.info(f"Running allocation for LKR {BUDGET_LKR:,.2f} budget across {len(potentials):,} outlets...")
    allocated_spends = allocate_budget_bisection(
        potentials=potentials,
        total_budget=BUDGET_LKR,
        alpha=ALPHA,
        beta=BETA,
        max_spend=MAX_SPEND_PER_OUTLET_LKR
    )
    
    wp_df["Trade_Spend_LKR"] = np.round(allocated_spends, 2)
    
    # 5. Compute Expected Lift and ROI
    # Lift = Potential_Gap * Alpha * log(1 + Spend/Beta)
    wp_df["Expected_Lift"] = np.round(
        wp_df["potential_gap"] * ALPHA * np.log1p(wp_df["Trade_Spend_LKR"] / BETA), 2
    )
    
    # ROI = Expected_Lift / Trade_Spend_LKR (volume lift in liters per LKR spent)
    wp_df["ROI"] = np.where(
        wp_df["Trade_Spend_LKR"] > 0,
        np.round(wp_df["Expected_Lift"] / wp_df["Trade_Spend_LKR"], 6),
        0.0
    )
    
    # 6. Format and save output
    output_cols = ["Outlet_ID", "Trade_Spend_LKR", "Expected_Lift", "ROI"]
    allocations_df = wp_df[output_cols].copy()
    
    # Sort by spend descending for readability
    allocations_df = allocations_df.sort_values("Trade_Spend_LKR", ascending=False)
    
    # Write to both target naming formats to ensure submission compatibility
    out_path_lower = OUTPUT / "ai_aces_budget_allocations.csv"
    out_path_upper = OUTPUT / "AI_ACES_budget_allocations.csv"
    
    allocations_df.to_csv(out_path_lower, index=False)
    allocations_df.to_csv(out_path_upper, index=False)
    
    logger.info(f"Saved budget allocations to: {out_path_lower}")
    
    # 7. Print summary statistics
    total_allocated = allocations_df["Trade_Spend_LKR"].sum()
    total_lift = allocations_df["Expected_Lift"].sum()
    active_allocations = (allocations_df["Trade_Spend_LKR"] > 0).sum()
    avg_roi = allocations_df.loc[allocations_df["Trade_Spend_LKR"] > 0, "ROI"].mean() if active_allocations > 0 else 0.0
    
    logger.info("\n" + "="*80 + "\nBUDGET OPTIMIZATION SUMMARY:\n" + "="*80)
    logger.info(f"Total Budget LKR:             {BUDGET_LKR:,.2f}")
    logger.info(f"Total Allocated LKR:          {total_allocated:,.2f} ({(total_allocated/BUDGET_LKR)*100:.2f}%)")
    logger.info(f"Total Incremental Lift (L):   {total_lift:,.2f} Liters")
    logger.info(f"Outlets receiving budget:     {active_allocations:,} / {len(wp_df):,} ({(active_allocations/len(wp_df))*100:.1f}%)")
    logger.info(f"Average Lift per LKR (ROI):   {avg_roi:.6f} Liters/LKR")
    logger.info(f"Maximum Spend on single outlet: LKR {allocations_df['Trade_Spend_LKR'].max():,.2f}")
    logger.info("="*80 + "\n")

if __name__ == "__main__":
    main()
