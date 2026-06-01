"""
DataStorm 2026 - Settings Routes
================================
Simple application settings for business-friendly defaults and alert preferences.
"""

from flask import Blueprint, render_template, request, redirect, url_for, flash, session

settings_bp = Blueprint("settings", __name__)

DEFAULT_SETTINGS = {
    "default_landing_page": "dashboard",
    "optimizer_default_budget": 5000000,
    "censoring_alert_threshold": 0.30,
    "export_format": "CSV",
    "notify_on_reports": False,
}

@settings_bp.route("/settings", methods=["GET", "POST"])
def view_settings():
    """Render the settings page and persist preferences in the user session."""
    if request.method == "POST":
        values = {
            "default_landing_page": request.form.get("default_landing_page", "dashboard"),
            "optimizer_default_budget": float(request.form.get("optimizer_default_budget", 5000000) or 5000000),
            "censoring_alert_threshold": float(request.form.get("censoring_alert_threshold", 0.30) or 0.30),
            "export_format": request.form.get("export_format", "CSV"),
            "notify_on_reports": request.form.get("notify_on_reports") == "on",
        }
        session["app_settings"] = values
        flash("Settings saved successfully.", "success")
        return redirect(url_for("settings.view_settings"))

    settings = session.get("app_settings", DEFAULT_SETTINGS.copy())
    return render_template(
        "settings.html",
        active_page="settings",
        settings=settings,
    )
