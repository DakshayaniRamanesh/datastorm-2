"""
DataStorm 2026 - AI Data Assistant & Reports Routes
===================================================
Controller routes for the local AI Chatbot (AI Data Assistant) and validation reports.
Supports document queries, programmatic statistics, file uploads, and Ollama integration.
"""

import csv
import json
from pathlib import Path
from flask import Blueprint, render_template, abort, request, jsonify, current_app
from werkzeug.utils import secure_filename

from app.services.instances import xai, assistant

ROOT = Path(__file__).resolve().parents[2]
UPLOAD_FOLDER = ROOT / "uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

xai_bp = Blueprint("xai", __name__)


@xai_bp.route("/xai")
def view_xai():
    """Render the AI Data Assistant Chat interface (replacing the old XAI dashboard)."""
    return render_template(
        "xai.html",
        active_page="xai"
    )


@xai_bp.route("/xai/upload", methods=["POST"])
def upload_file():
    """API endpoint to upload CSV, Excel, or PDF files for domain chatbot retrieval."""
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request."}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file."}), 400

    filename = secure_filename(file.filename)
    filepath = UPLOAD_FOLDER / filename
    
    try:
        file.save(filepath)
        # Process and ingest in assistant service
        status_msg = assistant.ingest_uploaded_file(filepath)
        return jsonify({"message": status_msg})
    except Exception as e:
        return jsonify({"error": f"Failed to upload and ingest file: {str(e)}"}), 500


@xai_bp.route("/xai/chat", methods=["POST"])
def chat():
    """API endpoint to receive messages, apply guardrails, perform RAG, and query the local LLM."""
    data = request.get_json() or {}
    message = data.get("message", "").strip()
    history = data.get("history", [])

    if not message:
        return jsonify({"error": "Empty message."}), 400

    # 1. Apply strict domain guardrails
    if not assistant.is_query_domain_restricted(message):
        return jsonify({
            "response": "I can only answer questions related to the uploaded datasets, predictions, analytics, and project documentation.",
            "tokens": {"input": 0, "output": 0, "total": 0}
        })

    # 2. Run Analytics Engine to programmatically calculate statistics (mean, correlation, etc.)
    analytics_info = assistant.run_statistics_engine(message)

    # 3. Retrieve relevant context chunks (FAISS or TF-IDF fallback)
    retrieved_context = assistant.retrieve_context(message)

    # 4. Generate LLM response using local model (Ollama)
    response_text, tokens = assistant.generate_chat_response(
        query=message,
        context=retrieved_context,
        analytics_info=analytics_info,
        history=history
    )

    return jsonify({
        "response": response_text,
        "tokens": tokens
    })


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
