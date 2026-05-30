"""
DataStorm 2026 - Optimizer Routes
=================================
Controller routes for the interactive Trade Spend Budget Allocator.
Enables business managers to test custom budgets, run SLSQP optimization,
and view expected volume lifts and ROI reports in real-time.
"""

from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
from app.services.instances import optimization

optimizer_bp = Blueprint("optimizer", __name__)

@optimizer_bp.route("/optimizer", methods=["GET"])
def view_optimizer():
    """Render the main budget optimization form and dashboard."""
    summary = optimization.get_current_allocations_summary()
    
    # Defaults for form
    default_budget = 5000000.0
    default_cap = 100000.0
    
    return render_template(
        "optimization.html",
        active_page="optimizer",
        summary=summary,
        budget=default_budget,
        cap=default_cap,
        results=None
    )

@optimizer_bp.route("/optimizer/allocate", methods=["POST"])
def run_allocation():
    """Execute optimization run with user parameters and reload dashboard."""
    try:
        budget = float(request.form.get("budget", 5000000.0))
        cap = float(request.form.get("cap", 100000.0))
        
        # Run optimization (commits to DB)
        results = optimization.run_optimization(budget, cap)
        
        if results.get("status") == "error":
            flash(results["message"], "danger")
            return redirect(url_for("optimizer.view_optimizer"))
            
        summary = optimization.get_current_allocations_summary()
        
        flash(f"Convex optimization complete: LKR {results['total_allocated_lkr']:,.2f} allocated across {results['funded_outlets']} outlets.", "success")
        
        return render_template(
            "optimization.html",
            active_page="optimizer",
            summary=summary,
            budget=budget,
            cap=cap,
            results=results
        )
    except Exception as e:
        flash(f"Optimization execution failed: {e}", "danger")
        return redirect(url_for("optimizer.view_optimizer"))

@optimizer_bp.route("/api/optimizer/allocate", methods=["POST"])
def api_run_allocation():
    """AJAX endpoint for running optimizer."""
    try:
        data = request.get_json() or {}
        budget = float(data.get("budget", 5000000.0))
        cap = float(data.get("cap", 100000.0))
        
        results = optimization.run_optimization(budget, cap)
        return jsonify(results)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400
