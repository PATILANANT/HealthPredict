"""Administrator dashboard: usage statistics, users/roles, datasets and retraining (FR-16)."""
from __future__ import annotations

import shutil
import subprocess
import sys
import threading
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from flask import (Blueprint, abort, current_app, flash, redirect, render_template, request,
                   url_for)
from flask_login import current_user
from sqlalchemy import func

from ml.config import DATA_FILES, REQUIRED_COLUMNS, ROOT
from ml.schemas import MODULES

from .extensions import db, registry
from .models import ROLES, Prediction, TrainingRun, User
from .utils import roles_required

bp = Blueprint("admin", __name__, url_prefix="/admin")


@bp.route("/")
@roles_required("admin")
def index():
    since = datetime.now() - timedelta(days=13)
    per_module = dict(db.session.query(Prediction.module, func.count(Prediction.id))
                      .group_by(Prediction.module).all())
    per_risk = dict(db.session.query(Prediction.risk_level, func.count(Prediction.id))
                    .group_by(Prediction.risk_level).all())
    daily_rows = (db.session.query(func.date(Prediction.created_at), func.count(Prediction.id))
                  .filter(Prediction.created_at >= since.replace(hour=0, minute=0))
                  .group_by(func.date(Prediction.created_at)).all())
    daily_map = {str(d): c for d, c in daily_rows}
    days = [(since + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(14)]
    stats = {
        "users": User.query.count(),
        "predictions": Prediction.query.count(),
        "high_risk": per_risk.get("High", 0),
        "api_calls": Prediction.query.filter_by(source="api").count(),
    }
    recent = Prediction.query.order_by(Prediction.created_at.desc()).limit(10).all()
    return render_template("admin/index.html", stats=stats, per_module=per_module,
                           per_risk=per_risk, days=days, daily=[daily_map.get(d, 0) for d in days],
                           recent=recent)


@bp.route("/users", methods=["GET", "POST"])
@roles_required("admin")
def users():
    if request.method == "POST":
        user = db.session.get(User, int(request.form["user_id"])) or abort(404)
        role = request.form.get("role")
        if role not in ROLES:
            abort(400)
        if user.id == current_user.id and role != "admin":
            flash("You cannot remove your own admin role.", "warning")
        else:
            user.role = role
            db.session.commit()
            flash(f"{user.email} is now '{role}'.", "success")
        return redirect(url_for("admin.users"))
    return render_template("admin/users.html", users=User.query.order_by(User.created_at).all(),
                           roles=ROLES)


@bp.route("/models")
@roles_required("admin")
def models():
    runs = TrainingRun.query.order_by(TrainingRun.started_at.desc()).limit(15).all()
    datasets = {}
    for m, path in DATA_FILES.items():
        p = Path(path)
        datasets[m] = {"path": p.name, "exists": p.exists(),
                       "size_kb": round(p.stat().st_size / 1024) if p.exists() else 0,
                       "modified": datetime.fromtimestamp(p.stat().st_mtime) if p.exists() else None}
    return render_template("admin/models.html", runs=runs, datasets=datasets,
                           reports={m: registry.metrics(m) for m in MODULES},
                           running={r.module for r in runs if r.status == "running"})


@bp.route("/models/<module>/upload", methods=["POST"])
@roles_required("admin")
def upload(module):
    if module not in MODULES:
        abort(404)
    file = request.files.get("dataset")
    if not file or not file.filename.lower().endswith(".csv"):
        flash("Please choose a .csv file.", "warning")
        return redirect(url_for("admin.models"))
    tmp = Path(current_app.instance_path) / f"upload_{module}.csv"
    file.save(tmp)
    try:
        cols = set(pd.read_csv(tmp, nrows=5).columns)
    except Exception as exc:  # noqa: BLE001
        tmp.unlink(missing_ok=True)
        flash(f"Could not read the CSV: {exc}", "danger")
        return redirect(url_for("admin.models"))
    missing = set(REQUIRED_COLUMNS[module]) - cols
    if missing:
        tmp.unlink(missing_ok=True)
        flash(f"Dataset rejected - missing columns: {', '.join(sorted(missing))}", "danger")
        return redirect(url_for("admin.models"))
    target = Path(DATA_FILES[module])
    if target.exists():
        backup = target.with_name(f"{target.stem}.backup-{datetime.now():%Y%m%d-%H%M%S}.csv")
        shutil.copy2(target, backup)
    shutil.move(str(tmp), target)
    flash(f"Dataset for {MODULES[module]['title']} updated. Click Retrain to use it.", "success")
    return redirect(url_for("admin.models"))


def _run_training(app, run_id: int, module: str, force: bool):
    cmd = [sys.executable, "-m", f"ml.{module}.train"] + (["--force"] if force else [])
    if module == "mental_health":
        cmd.append("--fast")
    try:
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=3600)
        output = proc.stdout + ("\n" + proc.stderr if proc.returncode else "")
        ok = proc.returncode == 0
    except Exception as exc:  # noqa: BLE001
        output, ok = str(exc), False
    with app.app_context():
        run = db.session.get(TrainingRun, run_id)
        run.status = "success" if ok else "failed"
        run.promoted = ok and "-> PROMOTED" in output
        run.log = output[-20000:]
        run.finished_at = datetime.now()
        db.session.commit()
        if run.promoted:
            registry.load()


@bp.route("/models/<module>/retrain", methods=["POST"])
@roles_required("admin")
def retrain(module):
    if module not in MODULES:
        abort(404)
    if TrainingRun.query.filter_by(module=module, status="running").first():
        flash("Training for this module is already running.", "warning")
        return redirect(url_for("admin.models"))
    run = TrainingRun(module=module, triggered_by=current_user.id)
    db.session.add(run)
    db.session.commit()
    app = current_app._get_current_object()
    threading.Thread(target=_run_training,
                     args=(app, run.id, module, bool(request.form.get("force"))),
                     daemon=True).start()
    flash(f"Retraining {MODULES[module]['title']} started. Refresh this page to see progress.", "info")
    return redirect(url_for("admin.models"))


@bp.route("/runs/<int:run_id>")
@roles_required("admin")
def run(run_id):
    return render_template("admin/run.html", run=db.session.get(TrainingRun, run_id) or abort(404))


@bp.route("/reload", methods=["POST"])
@roles_required("admin")
def reload_models():
    registry.load()
    flash("Models reloaded from disk.", "success")
    return redirect(url_for("admin.models"))
