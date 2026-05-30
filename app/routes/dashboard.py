"""
DataStorm 2026 - Dashboard Routes
=================================
Controller routes for the main SaaS dashboard.
Renders KPI statistics, top provinces, top distributors, and Leaflet heatmap points.
"""

import json
from flask import Blueprint, render_template, jsonify
from app.services.instances import prediction, spatial

dashboard_bp = Blueprint("dashboard", __name__)

@dashboard_bp.route("/")
@dashboard_bp.route("/dashboard")
def view_dashboard():
    """Render the main analytics executive dashboard."""
    # Fetch KPIs
    kpis = prediction.get_summary_kpis()
    
    # Calculate Gaps and Percentages
    tot_potential = kpis.get("total_predicted_potential", 0.0) or 0.0
    tot_hist = kpis.get("total_hist_median", 0.0) or 0.0
    kpis["total_opportunity_gap"] = round(tot_potential - tot_hist, 2)
    
    if tot_hist > 0:
        kpis["growth_potential_percent"] = round(((tot_potential - tot_hist) / tot_hist) * 100, 1)
    else:
        kpis["growth_potential_percent"] = 0.0

    # Fetch regional summaries
    provinces = prediction.get_province_distribution()
    distributors = prediction.get_distributor_distribution()
    
    # Fetch coordinate points for Leaflet heatmap (limit to 5000 points to keep page rendering snappy)
    map_points = spatial.get_all_outlet_map_points()
    
    return render_template(
        "dashboard.html",
        active_page="dashboard",
        kpis=kpis,
        provinces=provinces,
        distributors=distributors[:6],  # display top 6
        map_points_json=json.dumps(map_points)
    )

@dashboard_bp.route("/api/map-points")
def api_map_points():
    """Serve outlet coordinates for asynchronous Leaflet calls."""
    points = spatial.get_all_outlet_map_points()
    return jsonify(points)
