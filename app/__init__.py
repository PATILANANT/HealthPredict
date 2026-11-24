"""Health Prediction System - Flask application factory."""
from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, render_template

from ml.config import DATA_DIR, MODELS_DIR, ROOT
from ml.schemas import MODULES

from .extensions import csrf, db, login_manager, registry

DISCLAIMER = ("This tool is a screening and decision-support aid only. It is NOT a medical "
              "diagnosis. Please consult a qualified doctor about any health concern.")


def create_app(config: dict | None = None) -> Flask:
    app = Flask(__name__, instance_path=str(ROOT / "instance"))
    app.config.from_mapping(
        SECRET_KEY=os.environ.get("SECRET_KEY", "dev-only-change-this-secret"),
        SQLALCHEMY_DATABASE_URI=os.environ.get(
            "DATABASE_URL", "sqlite:///" + str(Path(app.instance_path) / "health.db")),
        MODELS_DIR=str(MODELS_DIR),
        DATA_DIR=str(DATA_DIR),
        MAX_CONTENT_LENGTH=60 * 1024 * 1024,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
    )
    if config:
        app.config.update(config)
    Path(app.instance_path).mkdir(parents=True, exist_ok=True)

    db.init_app(app)
    login_manager.init_app(app)
    csrf.init_app(app)
    registry.models_dir = Path(app.config["MODELS_DIR"])
    registry.load()

    from .admin import bp as admin_bp
    from .api import bp as api_bp
    from .auth import bp as auth_bp
    from .doctor import bp as doctor_bp
    from .main import bp as main_bp
    from .predict import bp as predict_bp
    from .report import bp as report_bp

    for bp in (main_bp, auth_bp, predict_bp, report_bp, admin_bp, doctor_bp, api_bp):
        app.register_blueprint(bp)
    csrf.exempt(api_bp)

    from .cli import register_cli
    register_cli(app)

    from ml.schemas import grouped_fields, range_status

    from .advice import advice_for

    @app.context_processor
    def inject_globals():
        return {"MODULES": MODULES, "DISCLAIMER": DISCLAIMER, "registry": registry,
                "grouped_fields": grouped_fields, "range_status": range_status,
                "advice_for": advice_for}

    @app.template_filter("num")
    def num(value):
        """58.0 -> 58, 27.456 -> 27.5; strings pass through."""
        if isinstance(value, float):
            return f"{value:.0f}" if value.is_integer() else f"{value:.1f}".rstrip("0").rstrip(".")
        return value

    @app.template_filter("pct")
    def pct(value, digits=1):
        return "-" if value is None else f"{float(value) * 100:.{digits}f}%"

    @app.errorhandler(403)
    def forbidden(_e):
        return render_template("error.html", code=403, message="You do not have access to this page."), 403

    @app.errorhandler(404)
    def not_found(_e):
        return render_template("error.html", code=404, message="Page not found."), 404

    with app.app_context():
        db.create_all()
    return app
