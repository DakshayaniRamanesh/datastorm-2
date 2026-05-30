"""
DataStorm 2026 - Optimization Service
=====================================
Wraps the convex budget allocation algorithm for interactive use in the Flask web app.
When run, it recalculates allocations based on custom inputs and commits the new
results to the SQLite 'allocations' table, dynamically updating the entire platform.
"""

import logging
import importlib
import numpy as np
from typing import Dict, Any, List, Optional

from app.services.db_service import DBService

budget_optimizer = importlib.import_module("pipeline.05_budget_optimizer")
allocate_budget_bisection = budget_optimizer.allocate_budget_bisection
derive_median_price_per_liter = budget_optimizer.derive_median_price_per_liter

logger = logging.getLogger("OptimizationService")

ALPHA = 0.15
BETA = 10000.0


class OptimizationService:
    def __init__(self, db_service: DBService):
        self.db = db_service
        self._price_per_liter: Optional[float] = None

    def _get_price_per_liter(self) -> float:
        if self._price_per_liter is None:
            self._price_per_liter = derive_median_price_per_liter()
        return self._price_per_liter

    def run_optimization(self, total_budget: float, max_spend_cap: float) -> Dict[str, Any]:
        """Perform budget optimization on Western Province outlets and update database records."""
        query = """
            SELECT Outlet_ID, hist_median_vol, Maximum_Monthly_Liters
            FROM outlets
            WHERE primary_dist LIKE 'DIST_W_%'
        """
        outlets = self.db.execute_query(query)
        if not outlets:
            return {"status": "error", "message": "No outlets found in Western Province."}

        N = len(outlets)
        outlet_ids = [o["Outlet_ID"] for o in outlets]
        hist_vols = np.array([o["hist_median_vol"] for o in outlets])
        predicted_potentials = np.array([o["Maximum_Monthly_Liters"] for o in outlets])
        potentials_gap = np.clip(predicted_potentials - hist_vols, 0.0, None)

        logger.info(f"Running interactive optimization: budget={total_budget}, cap={max_spend_cap} on {N} outlets...")
        spends = allocate_budget_bisection(
            potentials=potentials_gap,
            total_budget=total_budget,
            alpha=ALPHA,
            beta=BETA,
            max_spend=max_spend_cap,
        )

        lifts = np.round(potentials_gap * ALPHA * np.log1p(spends / BETA), 2)
        rois = np.where(spends > 0, np.round(lifts / (spends + 1e-9), 6), 0.0)

        price_per_liter = self._get_price_per_liter()
        revenue = np.round(lifts * price_per_liter, 2)
        revenue_roi = np.where(spends > 0, np.round(revenue / (spends + 1e-9), 4), 0.0)

        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM allocations")
            insert_data = [
                (
                    outlet_ids[i],
                    float(spends[i]),
                    float(lifts[i]),
                    float(rois[i]),
                    float(revenue[i]),
                    float(revenue_roi[i]),
                )
                for i in range(N)
                if spends[i] > 0
            ]
            cursor.executemany(
                """INSERT INTO allocations
                   (Outlet_ID, Trade_Spend_LKR, Expected_Lift, ROI, Expected_Revenue_LKR, Revenue_ROI)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                insert_data,
            )
            conn.commit()

        active_allocations = len(insert_data)
        total_allocated = float(np.sum(spends))
        total_lift = float(np.sum(lifts))
        total_revenue = float(np.sum(revenue))
        avg_roi = float(np.mean(rois[spends > 0])) if active_allocations > 0 else 0.0
        avg_rev_roi = float(np.mean(revenue_roi[spends > 0])) if active_allocations > 0 else 0.0

        details = []
        for i in range(N):
            if spends[i] > 0:
                details.append({
                    "Outlet_ID": outlet_ids[i],
                    "potential_gap": float(potentials_gap[i]),
                    "Trade_Spend_LKR": float(spends[i]),
                    "Expected_Lift": float(lifts[i]),
                    "ROI": float(rois[i]),
                    "Expected_Revenue_LKR": float(revenue[i]),
                    "Revenue_ROI": float(revenue_roi[i]),
                })

        details = sorted(details, key=lambda x: x["Trade_Spend_LKR"], reverse=True)

        return {
            "status": "success",
            "total_outlets_wp": N,
            "funded_outlets": active_allocations,
            "total_allocated_lkr": total_allocated,
            "total_expected_lift_liters": total_lift,
            "total_expected_revenue_lkr": total_revenue,
            "price_per_liter_lkr": price_per_liter,
            "average_roi": avg_roi,
            "average_revenue_roi": avg_rev_roi,
            "allocations_sample": details[:100],
            "total_funded_percent": round((active_allocations / N) * 100, 1),
        }

    def get_current_allocations_summary(self) -> Dict[str, Any]:
        """Fetch the currently active allocations summary KPIs from the DB."""
        query = """
            SELECT
                COUNT(*) as funded_count,
                SUM(Trade_Spend_LKR) as total_spend,
                SUM(Expected_Lift) as total_lift,
                SUM(Expected_Revenue_LKR) as total_revenue,
                AVG(ROI) as avg_roi,
                AVG(Revenue_ROI) as avg_revenue_roi
            FROM allocations
        """
        rows = self.db.execute_query(query)
        res = rows[0] if rows else {}
        return {
            "funded_count": res.get("funded_count", 0) or 0,
            "total_spend": res.get("total_spend", 0.0) or 0.0,
            "total_lift": res.get("total_lift", 0.0) or 0.0,
            "total_revenue": res.get("total_revenue", 0.0) or 0.0,
            "avg_roi": res.get("avg_roi", 0.0) or 0.0,
            "avg_revenue_roi": res.get("avg_revenue_roi", 0.0) or 0.0,
        }
