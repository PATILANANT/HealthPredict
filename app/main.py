from flask import Blueprint, abort, render_template
from flask_login import current_user, login_required

from ml.schemas import MODULES

from .extensions import registry
from .models import Prediction
from .utils import latest_per_module

bp = Blueprint("main", __name__)


def model_highlights() -> dict:
    """Headline numbers for the landing page, read from the saved metrics."""
    out = {"classifiers": 0}
    for m in MODULES:
        rep = registry.metrics(m)
        if not rep:
            continue
        out["classifiers"] += len(rep["models"])
        t = rep["models"][rep["best_model"]]["test"]
        out[m] = {"best": rep["best_model"], "test": t}
        if m == "mental_health" and rep.get("distress_binary_test"):
            out[m]["distress"] = rep["distress_binary_test"]
    return out


@bp.route("/")
def index():
    return render_template("index.html", hl=model_highlights())


@bp.route("/about")
def about():
    return render_template("about.html", hl=model_highlights())


@bp.route("/dashboard")
@login_required
def dashboard():
    history = current_user.predictions.order_by(Prediction.created_at.desc()).limit(50).all()
    
    ordered = list(reversed(history))
    trend = {"labels": [p.created_at.strftime("%d %b, %H:%M") for p in ordered],
             "series": {m: [round(p.probability * 100, 1) if p.module == m else None for p in ordered]
                        for m in MODULES}}
    counts = {"total": current_user.predictions.count(),
              "high": current_user.predictions.filter(Prediction.risk_level == "High").count()}
    return render_template("dashboard.html", history=history, latest=latest_per_module(current_user),
                           trend=trend, counts=counts)


@bp.route("/metrics/<module>")
@login_required
def metrics(module):
    if module not in MODULES:
        abort(404)
    return render_template("metrics.html", module=module, report=registry.metrics(module))
