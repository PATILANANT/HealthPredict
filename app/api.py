"""REST API v1 for web / mobile clients (interoperability NFR).

Authentication: send your personal key (see Profile page) in the `X-API-Key` header.

    GET  /api/v1/health
    GET  /api/v1/modules                -> input schema of each module
    GET  /api/v1/models                 -> best model + test metrics per module
    POST /api/v1/predict/<module>       -> JSON body with the module's fields
    GET  /api/v1/predictions            -> your last 50 predictions
"""
from functools import wraps

from flask import Blueprint, g, jsonify, request

from ml.schemas import MODULES, validate

from .extensions import db, registry
from .models import Prediction, User

bp = Blueprint("api", __name__, url_prefix="/api/v1")


def api_key_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        key = request.headers.get("X-API-Key", "")
        user = User.query.filter_by(api_key=key).first() if key else None
        if user is None:
            return jsonify(error="Missing or invalid X-API-Key header."), 401
        g.api_user = user
        return view(*args, **kwargs)
    return wrapped


@bp.get("/health")
def health():
    return jsonify(status="ok", models={m: registry.get(m) is not None for m in MODULES})


@bp.get("/modules")
def modules():
    return jsonify({k: {"title": v["title"], "fields": [
        {kk: (list(map(list, vv)) if kk == "options" else vv) for kk, vv in f.items()}
        for f in v["fields"]]} for k, v in MODULES.items()})


@bp.get("/models")
def models():
    out = {}
    for m in MODULES:
        rep = registry.metrics(m)
        if rep:
            best = rep["best_model"]
            out[m] = {"best_model": best, "version": rep["version"],
                      "trained_at": rep["trained_at"],
                      "test_metrics": {k: v for k, v in rep["models"][best]["test"].items()
                                       if k not in ("per_class", "confusion", "labels")}}
    return jsonify(out)


@bp.post("/predict/<module>")
@api_key_required
def predict(module):
    if module not in MODULES:
        return jsonify(error=f"Unknown module '{module}'.", modules=list(MODULES)), 404
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(error="Send a JSON object body."), 400
    values, errors = validate(module, data)
    if errors:
        return jsonify(error="Validation failed.", fields=errors), 422
    if registry.get(module) is None:
        return jsonify(error=registry.errors.get(module, "Model unavailable.")), 503
    result = registry.predict(module, values)
    pred = Prediction.from_result(g.api_user, module, values, result, source="api")
    db.session.add(pred)
    db.session.commit()
    return jsonify(prediction_id=pred.id, **result,
                   disclaimer="Screening aid only - not a medical diagnosis.")


@bp.get("/predictions")
@api_key_required
def predictions():
    rows = g.api_user.predictions.order_by(Prediction.created_at.desc()).limit(50).all()
    return jsonify([{"id": p.id, "module": p.module, "label": p.label,
                     "probability": p.probability, "risk_level": p.risk_level,
                     "created_at": p.created_at.isoformat(timespec="seconds")} for p in rows])
