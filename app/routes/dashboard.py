"""
DataStorm 2026 - Dashboard Routes
=================================
Executive command-center dashboard: KPIs, spatial heatmap, analytics panels.
"""

import json
from pathlib import Path
from typing import Any, Dict

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


PROVINCE_FILTERS = {
    "western": {
        "prefix": "DIST_W",
        "bounds": {"lat_min": 6.6, "lat_max": 7.4, "lon_min": 79.7, "lon_max": 80.2},
    },
    "central": {
        "prefix": "DIST_C",
        "bounds": {"lat_min": 6.4, "lat_max": 8.6, "lon_min": 80.0, "lon_max": 81.5},
    },
    "northwestern": {
        "prefix": "DIST_NW",
        "bounds": {"lat_min": 7.1, "lat_max": 8.6, "lon_min": 79.4, "lon_max": 80.7},
    },
    "southern": {
        "prefix": "DIST_S",
        "bounds": {"lat_min": 5.7, "lat_max": 7.2, "lon_min": 79.4, "lon_max": 81.0},
    },
}


def _filter_province_outlets(points, province_key):
    config = PROVINCE_FILTERS.get(province_key)
    if not config:
        return []

    filtered = []
    for p in points:
        dist = (p.get("primary_dist") or "").upper()
        if not dist.startswith(config["prefix"]):
            continue

        try:
            lat = float(p.get("Latitude") or 0)
            lon = float(p.get("Longitude") or 0)
        except (TypeError, ValueError):
            continue

        bounds = config["bounds"]
        if lat < bounds["lat_min"] or lat > bounds["lat_max"]:
            continue
        if lon < bounds["lon_min"] or lon > bounds["lon_max"]:
            continue

        filtered.append(p)
    return filtered


@dashboard_bp.route("/api/province-map-bounds")
def api_province_map_bounds():
    """Per-province bounding boxes derived from outlet coordinates."""
    return jsonify(spatial.get_province_bounds_from_outlets())


@dashboard_bp.route("/api/province-boundaries")
def api_province_boundaries():
    """GeoJSON boundaries for Western, Central, North Western, and Southern provinces."""
    files = {
        "DIST_W": "western.geojson",
        "DIST_C": "central.geojson",
        "DIST_NW": "Northwestern.geojson",
        "DIST_S": "southern.geojson",
    }
    payload = {}
    for prefix, filename in files.items():
        path = STATIC_DATA / filename
        if path.exists():
            payload[prefix] = json.loads(path.read_text(encoding="utf-8"))
        else:
            payload[prefix] = {"type": "FeatureCollection", "features": []}
    return jsonify(payload)


@dashboard_bp.route("/api/province-outlets")
def api_province_outlets():
    """Outlet points for all provinces using coordinates from input/outlet_coordinates.csv."""
    points = spatial.get_all_outlet_map_points_from_csv()
    output: Dict[str, Any] = {}
    for province_key, cfg in PROVINCE_FILTERS.items():
        output[cfg["prefix"]] = _filter_province_outlets(points, province_key)
    return jsonify(output)
