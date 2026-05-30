"""
DataStorm 2026 - Outlets Explorer & Detail Routes
=================================================
Controller routes for outlet search, filtering, pagination, and detail audits.
Integrates spatial, optimization, and XAI services into a single detail dashboard.
"""

from flask import Blueprint, render_template, request, abort, jsonify
from app.services.instances import prediction, spatial, xai

outlets_bp = Blueprint("outlets", __name__)

@outlets_bp.route("/outlets")
def view_outlets():
    """Render the paginated and filterable Outlet Explorer page."""
    page = request.args.get("page", 1, type=int)
    search_query = request.args.get("search", "").strip()
    outlet_type = request.args.get("type", "")
    outlet_size = request.args.get("size", "")
    distributor_id = request.args.get("distributor", "")
    sort_by = request.args.get("sort_by", "Outlet_ID")
    sort_dir = request.args.get("sort_dir", "ASC")

    # Fetch filter options
    filters = prediction.get_filter_options()

    # Query paginated outlets
    per_page = 20
    rows, total_count = prediction.get_outlets_paginated(
        page=page,
        per_page=per_page,
        search_query=search_query,
        outlet_type=outlet_type,
        outlet_size=outlet_size,
        distributor_id=distributor_id,
        sort_by=sort_by,
        sort_dir=sort_dir
    )

    # Compute page boundaries
    total_pages = max(1, (total_count + per_page - 1) // per_page)
    
    return render_template(
        "outlets.html",
        active_page="outlets",
        outlets=rows,
        page=page,
        total_pages=total_pages,
        total_count=total_count,
        search_query=search_query,
        selected_type=outlet_type,
        selected_size=outlet_size,
        selected_distributor=distributor_id,
        sort_by=sort_by,
        sort_dir=sort_dir,
        filter_options=filters
    )

@outlets_bp.route("/outlets/<outlet_id>")
def view_outlet_detail(outlet_id: str):
    """Render the executive deep-dive audit page for a specific outlet."""
    # Fetch core details
    outlet_data = prediction.get_outlet_detail(outlet_id)
    if not outlet_data:
        abort(404, description=f"Outlet ID {outlet_id} not found.")

    # Fetch sales history
    sales_history = prediction.get_outlet_sales_history(outlet_id)

    # Fetch spatial competitors
    competitors = spatial.get_nearby_competitors(outlet_id, radius_m=1500.0)

    # Fetch SHAP contributions
    explanation = xai.get_local_explanation(outlet_id)

    # Fetch Claude recommendation report
    ai_narrative = xai.get_claude_advisor_narrative(outlet_data)

    return render_template(
        "outlet_detail.html",
        active_page="outlets",
        outlet=outlet_data,
        history=sales_history,
        competitors=competitors,
        explanation=explanation,
        narrative=ai_narrative
    )

@outlets_bp.route("/api/outlets/<outlet_id>/sales-history")
def api_outlet_sales_history(outlet_id: str):
    """API endpoint returning sales history for client-side charting."""
    history = prediction.get_outlet_sales_history(outlet_id)
    return jsonify(history)
