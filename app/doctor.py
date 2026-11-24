"""Healthcare-professional view: read-only access to patients' screening results."""
from flask import Blueprint, abort, render_template, request
from sqlalchemy import func

from .extensions import db
from .models import Prediction, User
from .utils import latest_per_module, roles_required

bp = Blueprint("doctor", __name__, url_prefix="/patients")


@bp.route("/")
@roles_required("doctor", "admin")
def patients():
    q = request.args.get("q", "").strip()
    users = User.query.filter_by(role="user")
    if q:
        like = f"%{q}%"
        users = users.filter((User.name.ilike(like)) | (User.email.ilike(like)))
    users = users.order_by(User.name).all()
    high = dict(db.session.query(Prediction.user_id, func.count(Prediction.id))
                .filter(Prediction.risk_level == "High").group_by(Prediction.user_id).all())
    rows = [(u, latest_per_module(u), high.get(u.id, 0)) for u in users]
    return render_template("doctor/patients.html", rows=rows, q=q)


@bp.route("/<int:user_id>")
@roles_required("doctor", "admin")
def patient(user_id):
    user = db.session.get(User, user_id) or abort(404)
    history = user.predictions.order_by(Prediction.created_at.desc()).all()
    return render_template("doctor/patient.html", patient=user, history=history,
                           latest=latest_per_module(user))
