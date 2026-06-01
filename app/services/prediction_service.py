"""
DataStorm 2026 - Prediction Service
===================================
Fetches and aggregates outlet predictions, KPIs, lists, and details.
Maps distributor prefixes to Sri Lankan provinces:
  - DIST_W_ -> Western Province
  - DIST_C_ -> Central Province
  - DIST_S_ -> Southern Province
  - DIST_NW_ -> North Western Province
"""

import csv
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from app.services.db_service import DBService

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
VALIDATION_REPORT = ROOT_DIR / "output" / "validation_report.csv"
MODEL_BENCHMARK = ROOT_DIR / "output" / "model_benchmark_chronological.csv"

logger = logging.getLogger("PredictionService")

PROVINCE_MAP = {
    "DIST_W": "Western Province",
    "DIST_C": "Central Province",
    "DIST_S": "Southern Province",
    "DIST_NW": "North Western Province"
}

class PredictionService:
    def __init__(self, db_service: DBService):
        self.db = db_service

    def get_summary_kpis(self) -> Dict[str, Any]:
        """Fetch network-wide KPIs for the executive dashboard panel."""
        query = """
            SELECT 
                COUNT(o.Outlet_ID) as total_outlets,
                SUM(o.hist_median_vol) as total_hist_median,
                SUM(o.Maximum_Monthly_Liters) as total_predicted_potential,
                AVG(o.censoring_score) as avg_censoring_score,
                SUM(CASE WHEN o.censoring_score > 0.30 THEN 1 ELSE 0 END) as total_censored
            FROM outlets o
        """
        res = self.db.execute_query(query)
        kpis = res[0] if res else {}
        
        # Load budget spend lift if available
        try:
            alloc_query = """
                SELECT
                    SUM(Trade_Spend_LKR) as total_spend,
                    SUM(Expected_Lift) as total_lift,
                    SUM(Expected_Revenue_LKR) as total_revenue,
                    AVG(Revenue_ROI) as avg_revenue_roi
                FROM allocations
            """
            alloc_res = self.db.execute_query(alloc_query)
            if alloc_res and alloc_res[0]["total_spend"]:
                kpis["total_spend_allocated"] = alloc_res[0]["total_spend"]
                kpis["total_expected_lift"] = alloc_res[0]["total_lift"]
                kpis["total_expected_revenue"] = alloc_res[0].get("total_revenue") or 0.0
                kpis["avg_revenue_roi"] = alloc_res[0].get("avg_revenue_roi") or 0.0
            else:
                kpis["total_spend_allocated"] = 0.0
                kpis["total_expected_lift"] = 0.0
                kpis["total_expected_revenue"] = 0.0
                kpis["avg_revenue_roi"] = 0.0
        except Exception:
            kpis["total_spend_allocated"] = 0.0
            kpis["total_expected_lift"] = 0.0
            kpis["total_expected_revenue"] = 0.0
            kpis["avg_revenue_roi"] = 0.0
            
        return kpis

    def get_province_distribution(self) -> List[Dict[str, Any]]:
        """Group and aggregate historical vs predicted volume by province."""
        outlets = self.db.execute_query("""
            SELECT primary_dist, hist_median_vol, Maximum_Monthly_Liters 
            FROM outlets
        """)
        
        prov_data = {}
        for row in outlets:
            dist = row["primary_dist"]
            prefix = "_".join(dist.split("_")[:2]) if dist else "Unknown"
            province = PROVINCE_MAP.get(prefix, "Other Provinces")
            
            if province not in prov_data:
                prov_data[province] = {"province": province, "hist_volume": 0.0, "predicted_potential": 0.0, "outlets_count": 0}
                
            prov_data[province]["hist_volume"] += row["hist_median_vol"] or 0.0
            prov_data[province]["predicted_potential"] += row["Maximum_Monthly_Liters"] or 0.0
            prov_data[province]["outlets_count"] += 1
            
        # Compute gap and convert to list
        out_list = []
        for p in prov_data.values():
            p["hist_volume"] = round(p["hist_volume"], 2)
            p["predicted_potential"] = round(p["predicted_potential"], 2)
            p["opportunity_gap"] = round(p["predicted_potential"] - p["hist_volume"], 2)
            out_list.append(p)
            
        return sorted(out_list, key=lambda x: x["predicted_potential"], reverse=True)

    def get_distributor_distribution(self) -> List[Dict[str, Any]]:
        """Aggregate predictions and gaps at the Distributor level."""
        query = """
            SELECT 
                primary_dist as distributor_id,
                COUNT(Outlet_ID) as outlets_count,
                SUM(hist_median_vol) as hist_volume,
                SUM(Maximum_Monthly_Liters) as predicted_potential
            FROM outlets
            GROUP BY primary_dist
            ORDER BY predicted_potential DESC
        """
        rows = self.db.execute_query(query)
        for r in rows:
            r["hist_volume"] = round(r["hist_volume"], 2)
            r["predicted_potential"] = round(r["predicted_potential"], 2)
            r["opportunity_gap"] = round(r["predicted_potential"] - r["hist_volume"], 2)
        return rows

    def get_distributor_summary(self, distributor_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a synthetic summary for a single distributor."""
        query = """
            SELECT
                o.primary_dist as distributor_id,
                COUNT(o.Outlet_ID) as outlets_count,
                AVG(o.censoring_score) as avg_censoring_score,
                SUM(o.hist_median_vol) as hist_volume,
                SUM(o.Maximum_Monthly_Liters) as predicted_potential,
                SUM(COALESCE(a.Trade_Spend_LKR, 0.0)) as total_allocated_spend,
                SUM(COALESCE(a.Expected_Lift, 0.0)) as total_expected_lift
            FROM outlets o
            LEFT JOIN allocations a ON o.Outlet_ID = a.Outlet_ID
            WHERE o.primary_dist = ?
            GROUP BY o.primary_dist
        """
        rows = self.db.execute_query(query, (distributor_id,))
        if not rows:
            return None
        result = rows[0]
        result["hist_volume"] = round(result["hist_volume"], 2)
        result["predicted_potential"] = round(result["predicted_potential"], 2)
        result["opportunity_gap"] = round(result["predicted_potential"] - result["hist_volume"], 2)
        result["avg_censoring_score"] = round(result.get("avg_censoring_score") or 0.0, 3)
        result["total_allocated_spend"] = round(result.get("total_allocated_spend") or 0.0, 2)
        result["total_expected_lift"] = round(result.get("total_expected_lift") or 0.0, 2)
        return result

    def get_outlets_paginated(
        self,
        page: int = 1,
        per_page: int = 50,
        search_query: str = "",
        outlet_type: str = "",
        outlet_size: str = "",
        distributor_id: str = "",
        sort_by: str = "Outlet_ID",
        sort_dir: str = "ASC"
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Query outlets list with search, filter, sorting, and SQL limit pagination."""
        where_clauses = []
        params = []

        if search_query:
            where_clauses.append("(o.Outlet_ID LIKE ? OR o.primary_dist LIKE ?)")
            params.extend([f"%{search_query}%", f"%{search_query}%"])
        if outlet_type:
            where_clauses.append("o.Outlet_Type = ?")
            params.append(outlet_type)
        if outlet_size:
            where_clauses.append("o.Outlet_Size = ?")
            params.append(outlet_size)
        if distributor_id:
            where_clauses.append("o.primary_dist = ?")
            params.append(distributor_id)

        where_sql = ""
        if where_clauses:
            where_sql = "WHERE " + " AND ".join(where_clauses)

        # Validate sorting columns to prevent SQL injection
        allowed_sort_cols = {
            "Outlet_ID", "Outlet_Type", "Outlet_Size", "Cooler_Count",
            "hist_median_vol", "Maximum_Monthly_Liters", "censoring_score",
            "combined_catchment_score", "effective_catchment_score", "poi_catchment_score",
            "market_saturation_index", "opportunity_gap"
        }
        if sort_by not in allowed_sort_cols:
            sort_by = "Outlet_ID"
            
        sort_dir = "DESC" if sort_dir.upper() == "DESC" else "ASC"

        # Count query (alias matches data query WHERE clauses)
        count_query = f"SELECT COUNT(*) FROM outlets o {where_sql}"
        total_count = self.db.execute_scalar(count_query, tuple(params))

        # Data query — subqueries avoid JOIN column ambiguity on Outlet_ID
        offset = (page - 1) * per_page
        data_query = f"""
            SELECT o.*,
                   COALESCE((SELECT Trade_Spend_LKR FROM allocations a WHERE a.Outlet_ID = o.Outlet_ID LIMIT 1), 0.0) AS Trade_Spend_LKR,
                   COALESCE((SELECT Expected_Lift FROM allocations a WHERE a.Outlet_ID = o.Outlet_ID LIMIT 1), 0.0) AS Expected_Lift,
                   COALESCE((SELECT ROI FROM allocations a WHERE a.Outlet_ID = o.Outlet_ID LIMIT 1), 0.0) AS ROI,
                   COALESCE((SELECT Expected_Revenue_LKR FROM allocations a WHERE a.Outlet_ID = o.Outlet_ID LIMIT 1), 0.0) AS Expected_Revenue_LKR,
                   COALESCE((SELECT Revenue_ROI FROM allocations a WHERE a.Outlet_ID = o.Outlet_ID LIMIT 1), 0.0) AS Revenue_ROI
            FROM outlets o
            {where_sql}
            ORDER BY {"(o.Maximum_Monthly_Liters - o.hist_median_vol)" if sort_by == 'opportunity_gap' else f'o.{sort_by}'} {sort_dir}
            LIMIT ? OFFSET ?
        """
        data_params = params + [per_page, offset]
        rows = self.db.execute_query(data_query, tuple(data_params))

        # Add province label to each row helper
        for r in rows:
            dist = r["primary_dist"]
            prefix = "_".join(dist.split("_")[:2]) if dist else ""
            r["province"] = PROVINCE_MAP.get(prefix, "Other")

        return rows, total_count

    def get_outlet_detail(self, outlet_id: str) -> Optional[Dict[str, Any]]:
        """Get detail record for a single outlet, including its budget allocation details."""
        query = """
            SELECT o.*, 
                   COALESCE(a.Trade_Spend_LKR, 0.0) as Trade_Spend_LKR,
                   COALESCE(a.Expected_Lift, 0.0) as Expected_Lift,
                   COALESCE(a.ROI, 0.0) as ROI,
                   COALESCE(a.Expected_Revenue_LKR, 0.0) as Expected_Revenue_LKR,
                   COALESCE(a.Revenue_ROI, 0.0) as Revenue_ROI
            FROM outlets o
            LEFT JOIN allocations a ON o.Outlet_ID = a.Outlet_ID
            WHERE o.Outlet_ID = ?
        """
        rows = self.db.execute_query(query, (outlet_id,))
        if not rows:
            return None
            
        r = rows[0]
        dist = r["primary_dist"]
        prefix = "_".join(dist.split("_")[:2]) if dist else ""
        r["province"] = PROVINCE_MAP.get(prefix, "Other")
        return r

    def get_outlet_sales_history(self, outlet_id: str) -> List[Dict[str, Any]]:
        """Fetch chronologically ordered monthly sales volume history for plotting."""
        query = """
            SELECT Year, Month, monthly_volume as volume, total_revenue as revenue, sku_count, txn_count
            FROM transactions
            WHERE Outlet_ID = ?
            ORDER BY Year ASC, Month ASC
        """
        return self.db.execute_query(query, (outlet_id,))

    def get_filter_options(self) -> Dict[str, List[str]]:
        """Fetch unique values for categories, sizes, and distributors for selector dropdowns."""
        types = [r["Outlet_Type"] for r in self.db.execute_query("SELECT DISTINCT Outlet_Type FROM outlets WHERE Outlet_Type IS NOT NULL ORDER BY Outlet_Type")]
        sizes = [r["Outlet_Size"] for r in self.db.execute_query("SELECT DISTINCT Outlet_Size FROM outlets WHERE Outlet_Size IS NOT NULL ORDER BY Outlet_Size")]
        dists = [r["primary_dist"] for r in self.db.execute_query("SELECT DISTINCT primary_dist FROM outlets WHERE primary_dist IS NOT NULL ORDER BY primary_dist")]
        return {"types": types, "sizes": sizes, "distributors": dists}

    def get_top_opportunity_outlets(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Top outlets ranked by latent opportunity gap (predicted minus historical)."""
        rows = self.db.execute_query(
            """
            SELECT Outlet_ID, primary_dist, Outlet_Type, Outlet_Size,
                   hist_median_vol, Maximum_Monthly_Liters, censoring_score,
                   (Maximum_Monthly_Liters - hist_median_vol) AS opportunity_gap
            FROM outlets
            ORDER BY opportunity_gap DESC
            LIMIT ?
            """,
            (limit,),
        )
        for r in rows:
            dist = r.get("primary_dist") or ""
            prefix = "_".join(dist.split("_")[:2])
            r["province"] = PROVINCE_MAP.get(prefix, "Other")
            hist = r.get("hist_median_vol") or 0.0
            pot = r.get("Maximum_Monthly_Liters") or 0.0
            r["uplift_pct"] = round(((pot - hist) / hist * 100), 1) if hist > 0 else 0.0
        return rows

    def get_budget_recommendations(self, limit: int = 12) -> List[Dict[str, Any]]:
        """Top trade-spend allocation recommendations from the optimizer."""
        try:
            rows = self.db.execute_query(
                """
                SELECT a.Outlet_ID, o.primary_dist, o.Outlet_Type,
                       a.Trade_Spend_LKR, a.Expected_Lift, a.Revenue_ROI
                FROM allocations a
                JOIN outlets o ON o.Outlet_ID = a.Outlet_ID
                WHERE a.Trade_Spend_LKR > 0
                ORDER BY a.Expected_Lift DESC
                LIMIT ?
                """,
                (limit,),
            )
            return rows
        except Exception:
            return []

    def get_potential_distribution(self) -> List[Dict[str, Any]]:
        """Histogram buckets for predicted monthly potential (liters)."""
        rows = self.db.execute_query(
            """
            SELECT
                CASE
                    WHEN Maximum_Monthly_Liters < 300 THEN '0-300 L'
                    WHEN Maximum_Monthly_Liters < 600 THEN '300-600 L'
                    WHEN Maximum_Monthly_Liters < 900 THEN '600-900 L'
                    WHEN Maximum_Monthly_Liters < 1200 THEN '900-1200 L'
                    WHEN Maximum_Monthly_Liters < 1500 THEN '1200-1500 L'
                    ELSE '1500+ L'
                END AS bucket,
                COUNT(*) AS outlets
            FROM outlets
            GROUP BY bucket
            ORDER BY MIN(Maximum_Monthly_Liters)
            """
        )
        return rows

    def get_outlet_segmentation(self) -> List[Dict[str, Any]]:
        """Aggregate potential and historical volume by outlet type."""
        rows = self.db.execute_query(
            """
            SELECT Outlet_Type,
                   COUNT(*) AS outlets,
                   ROUND(SUM(hist_median_vol), 1) AS hist_volume,
                   ROUND(SUM(Maximum_Monthly_Liters), 1) AS predicted_potential
            FROM outlets
            WHERE Outlet_Type IS NOT NULL
            GROUP BY Outlet_Type
            ORDER BY predicted_potential DESC
            """
        )
        for r in rows:
            r["opportunity_gap"] = round(
                (r.get("predicted_potential") or 0) - (r.get("hist_volume") or 0), 1
            )
        return rows

    def get_censoring_distribution(self) -> List[Dict[str, Any]]:
        """Distribution of demand-censoring scores across the network."""
        return self.db.execute_query(
            """
            SELECT
                CASE
                    WHEN censoring_score < 0.15 THEN 'Low (<0.15)'
                    WHEN censoring_score < 0.30 THEN 'Moderate (0.15–0.30)'
                    WHEN censoring_score < 0.50 THEN 'Elevated (0.30–0.50)'
                    ELSE 'Severe (≥0.50)'
                END AS band,
                COUNT(*) AS outlets
            FROM outlets
            GROUP BY band
            ORDER BY MIN(censoring_score)
            """
        )

    def get_active_distributors_count(self) -> int:
        return int(
            self.db.execute_scalar(
                "SELECT COUNT(DISTINCT primary_dist) FROM outlets WHERE primary_dist IS NOT NULL"
            )
            or 0
        )

    def get_executive_dashboard_kpis(self) -> Dict[str, Any]:
        """KPI strip: outlets, avg potential, high-potential count, severe censoring %."""
        row = self.db.execute_query(
            """
            SELECT
                COUNT(*) AS total_outlets,
                AVG(Maximum_Monthly_Liters) AS avg_potential,
                SUM(CASE WHEN Maximum_Monthly_Liters >= 900 THEN 1 ELSE 0 END) AS high_potential_outlets,
                SUM(CASE WHEN censoring_score >= 0.50 THEN 1 ELSE 0 END) AS severe_censored,
                SUM(CASE WHEN censoring_score < 0.15 THEN 1 ELSE 0 END) AS low_risk_outlets
            FROM outlets
            """
        )
        if not row:
            return {
                "total_outlets": 0,
                "avg_potential": 0.0,
                "high_potential_outlets": 0,
                "severe_censoring_pct": 0.0,
                "low_risk_pct": 0.0,
            }
        r = row[0]
        total = int(r.get("total_outlets") or 0)
        severe = int(r.get("severe_censored") or 0)
        low = int(r.get("low_risk_outlets") or 0)
        return {
            "total_outlets": total,
            "avg_potential": round(float(r.get("avg_potential") or 0), 0),
            "high_potential_outlets": int(r.get("high_potential_outlets") or 0),
            "severe_censoring_pct": round((severe / total) * 100, 1) if total else 0.0,
            "low_risk_pct": round((low / total) * 100, 1) if total else 0.0,
        }

    def get_avg_potential_uplift_pct(self) -> float:
        row = self.db.execute_query(
            """
            SELECT AVG(
                CASE WHEN hist_median_vol > 0
                THEN (Maximum_Monthly_Liters - hist_median_vol) / hist_median_vol * 100
                ELSE NULL END
            ) AS avg_uplift
            FROM outlets
            """
        )
        return round((row[0]["avg_uplift"] or 0.0), 1) if row else 0.0

    def get_model_performance_summary(self) -> Dict[str, Any]:
        """Load walk-forward validation metrics for the primary heuristic model."""
        summary = {
            "model_name": "Heuristic Latent (Two-Regime)",
            "mae": None,
            "rmse": None,
            "mape": None,
            "r2": None,
            "accuracy_pct": None,
        }
        if MODEL_BENCHMARK.exists():
            with MODEL_BENCHMARK.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if "Heuristic" in row.get("Model", ""):
                        summary["mae"] = round(float(row["MAE"]), 1)
                        summary["rmse"] = round(float(row["RMSE"]), 1)
                        summary["mape"] = round(float(row["MAPE_%"]), 1)
                        summary["r2"] = round(float(row["R2"]), 3)
                        summary["accuracy_pct"] = round(max(0.0, float(row["R2"]) * 100), 1)
                        break
        if summary["r2"] is None and VALIDATION_REPORT.exists():
            r2_vals = []
            with VALIDATION_REPORT.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row.get("Model") == "Heuristic_Latent_TwoRegime" and row.get("R2"):
                        try:
                            r2_vals.append(float(row["R2"]))
                        except ValueError:
                            pass
            if r2_vals:
                mean_r2 = sum(r2_vals) / len(r2_vals)
                summary["r2"] = round(mean_r2, 3)
                summary["accuracy_pct"] = round(max(0.0, mean_r2 * 100), 1)
        return summary
