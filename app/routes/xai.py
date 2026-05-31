"""
DataStorm 2026 - XAI & Reports Routes
=====================================
Controller routes for Explainable AI (SHAP global values) and validation reports.
Reads validation_report.csv and renders chronological holdout comparison tables.
"""

import csv
import json
from pathlib import Path

from flask import Blueprint, render_template, abort
from app.services.instances import xai

ROOT = Path(__file__).resolve().parents[2]

xai_bp = Blueprint("xai", __name__)

@xai_bp.route("/xai")
def view_xai():
    """Render global SHAP explainability summaries."""
    importance = xai.get_global_importance()
    top_importance = importance[:15]
    explanation_type = xai.get_explanation_type()

    return render_template(
        "xai.html",
        active_page="xai",
        importance=top_importance,
        explanation_type=explanation_type,
    )

@xai_bp.route("/reports")
def view_reports():
    """Read validation_report.csv and display walk-forward holdout metrics."""
    report_path = xai.shap_path.parent.parent.parent / "output" / "validation_report.csv"
    
    if not report_path.exists():
        return render_template(
            "reports.html",
            active_page="reports",
            report_rows=None,
            averages=None
        )

    def safe_float(val):
        try:
            return float(val) if val and val.strip() != "" else 0.0
        except (ValueError, TypeError):
            return 0.0

    rows = []
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append({
                    "fold": row.get("fold"),
                    "period": row.get("period", row.get("val_window", "N/A")),
                    "model": row.get("Model", row.get("fold", "N/A")),
                    "mae": safe_float(row.get("MAE")),
                    "rmse": safe_float(row.get("RMSE")),
                    "mape": safe_float(row.get("MAPE_%")),
                    "r2": safe_float(row.get("R2"))
                })
    except Exception as e:
        abort(500, description=f"Error reading validation report: {e}")

    # Calculate averages per model type
    model_groups = {}
    for r in rows:
        m = r["model"]
        if m not in model_groups:
            model_groups[m] = {"mae": [], "rmse": [], "mape": [], "r2": []}
        model_groups[m]["mae"].append(r["mae"])
        model_groups[m]["rmse"].append(r["rmse"])
        model_groups[m]["mape"].append(r["mape"])
        model_groups[m]["r2"].append(r["r2"])

    averages = []
    for model, vals in model_groups.items():
        averages.append({
            "model": model,
            "mae": round(sum(vals["mae"]) / len(vals["mae"]), 2),
            "rmse": round(sum(vals["rmse"]) / len(vals["rmse"]), 2),
            "mape": round(sum(vals["mape"]) / len(vals["mape"]), 2),
            "r2": round(sum(vals["r2"]) / len(vals["r2"]), 4)
        })

    ceiling_summary = None
    ceiling_blend_comparison = None
    for path in (
        ROOT / "samples" / "ceiling_validation_summary.json",
        ROOT / "output" / "ceiling_validation_summary.json",
    ):
        if path.exists():
            try:
                with open(path, encoding="utf-8") as f:
                    ceiling_summary = json.load(f)
                break
            except (json.JSONDecodeError, OSError):
                pass

    for path in (
        ROOT / "samples" / "ceiling_blend_comparison.json",
        ROOT / "output" / "ceiling_blend_comparison.json",
    ):
        if path.exists():
            try:
                with open(path, encoding="utf-8") as f:
                    ceiling_blend_comparison = json.load(f)
                break
            except (json.JSONDecodeError, OSError):
                pass

    return render_template(
        "reports.html",
        active_page="reports",
        report_rows=rows,
        averages=averages,
        ceiling_summary=ceiling_summary,
        ceiling_blend_comparison=ceiling_blend_comparison,
    )
