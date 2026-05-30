"""
DataStorm 2026 - Flask Entrypoint
=================================
Configures and spins up the SaaS analytics web dashboard.
Performs automatic database compilation on startup if needed.
"""

import sys
import logging
from flask import Flask, redirect, url_for, send_from_directory, render_template
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

try:
    from app.services.instances import db
    from app.routes.dashboard import dashboard_bp
    from app.routes.outlets import outlets_bp
    from app.routes.optimizer import optimizer_bp
    from app.routes.xai import xai_bp
except ModuleNotFoundError:
    from services.instances import db
    from routes.dashboard import dashboard_bp
    from routes.outlets import outlets_bp
    from routes.optimizer import optimizer_bp
    from routes.xai import xai_bp

def create_app() -> Flask:
    """Application factory for the Flask server."""
    app = Flask(__name__)
    
    # Simple configuration (harkened for security)
    import secrets
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", secrets.token_hex(32))
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    
    # 1. Register blueprints
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(outlets_bp)
    app.register_blueprint(optimizer_bp)
    app.register_blueprint(xai_bp)
    
    # Serve generated output assets
    @app.route("/output/<filename>")
    def serve_output(filename):
        return send_from_directory(str(ROOT_DIR / "output"), filename)

    # Root redirect
    @app.route("/")
    @app.route("/index")
    def index():
        return redirect(url_for("dashboard.view_dashboard"))
        
    @app.errorhandler(404)
    def page_not_found(e):
        return render_template("404.html", description=e.description), 404
        
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
    logger.info("Starting Flask dev server...")
    app.run(host="0.0.0.0", port=5000, debug=os.environ.get("FLASK_DEBUG", "0") == "1")
