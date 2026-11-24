"""Combined health-risk report: HTML view and PDF download (FR-13)."""
from datetime import datetime

from flask import Blueprint, Response, abort, render_template
from flask_login import current_user, login_required

from .extensions import db
from .models import User
from .pdf import build_report
from .predict import get_visible_prediction
from .utils import latest_per_module

bp = Blueprint("report", __name__, url_prefix="/report")


def _pdf_response(data: bytes, filename: str) -> Response:
    return Response(data, mimetype="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@bp.route("/")
@login_required
def combined():
    return render_template("report.html", latest=latest_per_module(current_user),
                           owner=current_user, now=datetime.now())


@bp.route("/pdf")
@login_required
def combined_pdf():
    from . import DISCLAIMER
    latest = list(latest_per_module(current_user).values())
    if not latest:
        abort(404)
    return _pdf_response(build_report(current_user, latest, DISCLAIMER), "health_report.pdf")


@bp.route("/pdf/user/<int:user_id>")
@login_required
def user_pdf(user_id):
    from . import DISCLAIMER
    if not current_user.is_staff and current_user.id != user_id:
        abort(403)
    user = db.session.get(User, user_id) or abort(404)
    latest = list(latest_per_module(user).values())
    if not latest:
        abort(404)
    return _pdf_response(build_report(user, latest, DISCLAIMER), f"health_report_{user_id}.pdf")


@bp.route("/pdf/<int:pred_id>")
@login_required
def single_pdf(pred_id):
    from . import DISCLAIMER
    pred = get_visible_prediction(pred_id)
    return _pdf_response(build_report(pred.user, [pred], DISCLAIMER), f"prediction_{pred_id}.pdf")
