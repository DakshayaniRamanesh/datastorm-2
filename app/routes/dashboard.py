"""
DataStorm 2026 - Dashboard Routes
=================================
Executive command-center dashboard: KPIs, spatial heatmap, analytics panels.
"""

import json
from pathlib import Path

from flask import Blueprint, render_template, jsonify

from app.services.instances import prediction, spatial, xai

dashboard_bp = Blueprint("dashboard", __name__)

STATIC_DATA = Path(__file__).parent.parent / "static" / "data"


@dashboard_bp.route("/dashboard")
def view_dashboard():
    """Render the executive AI analytics command center."""
    exec_kpis = prediction.get_executive_dashboard_kpis()
    censoring_dist = prediction.get_censoring_distribution()

    return render_template(
        "dashboard.html",
        active_page="dashboard",
        exec_kpis=exec_kpis,
        provinces=prediction.get_province_distribution(),
        top_outlets=prediction.get_top_opportunity_outlets(20),
        budget_recs=prediction.get_budget_recommendations(12),
        potential_dist=prediction.get_potential_distribution(),
        segmentation=prediction.get_outlet_segmentation(),
        censoring_dist=censoring_dist,
        shap_importance=xai.get_global_importance()[:10],
    )


@dashboard_bp.route("/api/map-points")
def api_map_points():
    """Outlet coordinates for the 4-province spatial heatmap."""
    points = spatial.get_all_outlet_map_points(four_provinces_only=True)
    return jsonify(points)


@dashboard_bp.route("/api/province-map-bounds")
def api_province_map_bounds():
    """Per-province bounding boxes derived from outlet coordinates."""
    return jsonify(spatial.get_province_bounds_from_outlets())


@dashboard_bp.route("/api/province-boundaries")
def api_province_boundaries():
    """GeoJSON for Western, Central, North Western, and Southern provinces."""
    path = STATIC_DATA / "lk_provinces_4.geojson"
    if not path.exists():
        return jsonify({"type": "FeatureCollection", "features": []})
    return jsonify(json.loads(path.read_text(encoding="utf-8")))


@dashboard_bp.route("/api/western-outlets")
def api_western_outlets():
    """Outlet points for Western Province only with strict bounds validation."""
    western_bounds = {"lat_min": 6.6, "lat_max": 7.4, "lon_min": 79.7, "lon_max": 80.2}
    points = spatial.get_all_outlet_map_points(four_provinces_only=False)
    
    western = []
    for p in points:
        dist = (p.get("primary_dist") or "").upper()
        if not dist.startswith("DIST_W"):
            continue
        
        lat = float(p.get("Latitude") or 0)
        lon = float(p.get("Longitude") or 0)
        
        if lat < western_bounds["lat_min"] or lat > western_bounds["lat_max"]:
            continue
        if lon < western_bounds["lon_min"] or lon > western_bounds["lon_max"]:
            continue
        
        western.append(p)
    
    return jsonify(western)
