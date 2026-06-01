"""
DataStorm 2026 - Distributor Intelligence Routes
================================================
Controller routes for distributor-level performance, opportunity, and allocation insights.
"""

from flask import Blueprint, render_template, request, abort
from app.services.instances import prediction

distributor_bp = Blueprint("distributor", __name__)

@distributor_bp.route("/distributors")
def view_distributors():
    """Render the distributor-level summary and ranking page."""
    distributors = prediction.get_distributor_distribution()
    total_outlets = sum(d.get("outlets_count", 0) for d in distributors)
    total_gap = sum(d.get("opportunity_gap", 0.0) for d in distributors)
    return render_template(
        "distributors.html",
        active_page="distributors",
        distributors=distributors,
        total_outlets=total_outlets,
        total_gap=total_gap,
    )

@distributor_bp.route("/distributors/<distributor_id>")
def view_distributor_detail(distributor_id: str):
    """Render detailed metrics and outlet-level analytics for a distributor."""
    page = request.args.get("page", 1, type=int)
    per_page = 20

    summary = prediction.get_distributor_summary(distributor_id)
    if not summary:
        abort(404, description=f"Distributor {distributor_id} not found.")

    outlets, total_count = prediction.get_outlets_paginated(
        page=page,
        per_page=per_page,
        distributor_id=distributor_id,
        sort_by="opportunity_gap",
        sort_dir="DESC",
    )

    total_pages = max(1, (total_count + per_page - 1) // per_page)

    return render_template(
        "distributor_detail.html",
        active_page="distributors",
        distributor=summary,
        outlets=outlets,
        page=page,
        total_pages=total_pages,
        total_count=total_count,
    )
