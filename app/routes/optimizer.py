"""
DataStorm 2026 - Optimizer Routes
=================================
Budget allocation driven by outlet opportunity gap (predicted minus historical).
"""

from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
from app.services.instances import optimization

optimizer_bp = Blueprint("optimizer", __name__)


@optimizer_bp.route("/optimizer", methods=["GET"])
def view_optimizer():
    """Render the budget optimization form and summary."""
    return render_template(
        "optimization.html",
        active_page="optimizer",
        summary=optimization.get_current_allocations_summary(),
        budget=5_000_000.0,
        results=None,
    )


@optimizer_bp.route("/optimizer/allocate", methods=["POST"])
def run_allocation():
    """Execute gap-weighted budget allocation."""
    try:
        budget = float(request.form.get("budget", 5_000_000.0))
        results = optimization.run_optimization(budget)

        if results.get("status") == "error":
            flash(results["message"], "danger")
            return redirect(url_for("optimizer.view_optimizer"))

        summary = optimization.get_current_allocations_summary()
        flash(
            f"Allocation complete: LKR {results['total_allocated_lkr']:,.2f} across "
            f"{results['funded_outlets']} outlets (weighted by opportunity gap).",
            "success",
        )
        return render_template(
            "optimization.html",
            active_page="optimizer",
            summary=summary,
            budget=budget,
            results=results,
        )
    except Exception as e:
        flash(f"Optimization failed: {e}", "danger")
        return redirect(url_for("optimizer.view_optimizer"))


@optimizer_bp.route("/api/optimizer/allocate", methods=["POST"])
def api_run_allocation():
    """AJAX endpoint for running optimizer."""
    try:
        data = request.get_json() or {}
        budget = float(data.get("budget", 5_000_000.0))
        return jsonify(optimization.run_optimization(budget))
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400
