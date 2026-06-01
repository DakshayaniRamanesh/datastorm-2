"""
DataStorm 2026 - Flask Entrypoint
=================================
Configures and spins up the SaaS analytics web dashboard.
Performs automatic database compilation on startup if needed.
"""

import sys
import logging
from flask import Flask, redirect, url_for, send_from_directory, render_template, session
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("App")

# Append parent dir for relative imports (insert at 0 and filter CWD to prevent shadowing)
import os
ROOT_DIR = Path(__file__).parent.parent
current_dir = Path(__file__).parent.resolve()

cleaned_path = []
for p in sys.path:
    resolved_p = Path(os.getcwd() if not p else p).resolve()
    if resolved_p != current_dir:
        cleaned_path.append(p)
sys.path = cleaned_path

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from load_env import load_project_env
load_project_env()

try:
    from app.services.instances import db
    from app.routes.dashboard import dashboard_bp
    from app.routes.outlets import outlets_bp
    from app.routes.optimizer import optimizer_bp
    from app.routes.xai import xai_bp
    from app.routes.distributor import distributor_bp
    from app.routes.settings import settings_bp
except ModuleNotFoundError:
    from services.instances import db
    from routes.dashboard import dashboard_bp
    from routes.outlets import outlets_bp
    from routes.optimizer import optimizer_bp
    from routes.xai import xai_bp
    from routes.distributor import distributor_bp
    from routes.settings import settings_bp


def create_app() -> Flask:
    """Application factory for the Flask server."""
    app_dir = Path(__file__).parent
    app = Flask(
        __name__,
        template_folder=str(app_dir / "templates"),
        static_folder=str(app_dir / "static"),
    )
    
    # Simple configuration (harkened for security)
    import secrets
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", secrets.token_hex(32))
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    
    # 1. Register blueprints
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(outlets_bp)
    app.register_blueprint(distributor_bp)
    app.register_blueprint(optimizer_bp)
    app.register_blueprint(xai_bp)
    app.register_blueprint(settings_bp)
    
    # Serve generated output assets
    @app.route("/output/<filename>")
    def serve_output(filename):
        return send_from_directory(str(ROOT_DIR / "output"), filename)

    # Root redirect
    @app.route("/")
    @app.route("/index")
    def index():
        landing_page = session.get("app_settings", {}).get("default_landing_page", "dashboard")
        target = {
            "dashboard": "dashboard.view_dashboard",
            "outlets": "outlets.view_outlets",
            "distributors": "distributor.view_distributors",
            "optimizer": "optimizer.view_optimizer",
            "xai": "xai.view_xai",
        }.get(landing_page, "dashboard.view_dashboard")
        return redirect(url_for(target))
        
    @app.errorhandler(404)
    def page_not_found(e):
        return render_template("404.html", description=e.description), 404

    @app.errorhandler(500)
    def internal_error(e):
        logger.exception("Internal server error")
        return "Internal Server Error — check the terminal for the traceback.", 500
        
    # Inject active_page helper to templates
    @app.context_processor
    def inject_helpers():
        return dict(zip=zip, round=round, int=int, len=len)

    # 2. Compile SQLite Database on start
    try:
        db.compile_db(force=False)
    except Exception as e:
        logger.error(f"Failed to auto-compile SQLite database at startup: {e}")
        
    return app

app = create_app()

if __name__ == "__main__":
    import socket

    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"

    # Skip in werkzeug reloader child (parent already holds the port)
    if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                logger.error(
                    "Port %s is already in use. Stop other Flask instances first "
                    "(Ctrl+C in their terminal, or: Get-Process python | Stop-Process -Force).",
                    port,
                )
                sys.exit(1)

    logger.info("Starting Flask dev server on port %s...", port)
    app.run(host="0.0.0.0", port=port, debug=debug, use_reloader=debug)
