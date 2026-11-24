"""Prediction forms and results for the three modules (FR-02 ... FR-09, FR-14, FR-15)."""
from flask import Blueprint, abort, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required

from ml.schemas import MODULES, validate

from .extensions import db, registry
from .models import Prediction

bp = Blueprint("predict", __name__)


@bp.route("/predict/<module>", methods=["GET", "POST"])
@login_required
def form(module):
    if module not in MODULES:
        abort(404)
    errors, values = {}, {}
    if request.method == "POST":
        values = request.form.to_dict()
        clean, errors = validate(module, request.form)
        if not errors:
            if registry.get(module) is None:
                flash(registry.errors.get(module, "Model unavailable."), "danger")
            else:
                result = registry.predict(module, clean)
                pred = Prediction.from_result(current_user, module, clean, result)
                db.session.add(pred)
                db.session.commit()
                return redirect(url_for("predict.result", pred_id=pred.id))
        else:
            flash("Please correct the highlighted fields.", "warning")
    return render_template("predict/form.html", module=module, spec=MODULES[module],
                           errors=errors, values=values)


def get_visible_prediction(pred_id: int) -> Prediction:
    pred = db.session.get(Prediction, pred_id)
    if pred is None:
        abort(404)
    if pred.user_id != current_user.id and not current_user.is_staff:
        abort(403)
    return pred


@bp.route("/result/<int:pred_id>")
@login_required
def result(pred_id):
    pred = get_visible_prediction(pred_id)
    return render_template("predict/result.html", pred=pred, spec=MODULES[pred.module])
