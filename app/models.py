"""Database tables: users (D1), prediction log (FR-15) and model training runs (FR-16)."""
from __future__ import annotations

import json
import secrets
from datetime import datetime

from flask_login import UserMixin
from werkzeug.security import check_password_hash, generate_password_hash

from .extensions import db, login_manager

ROLES = ("user", "doctor", "admin")


def _now():
    return datetime.now()


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), nullable=False, default="user")
    api_key = db.Column(db.String(64), unique=True, index=True,
                        default=lambda: secrets.token_hex(20))
    created_at = db.Column(db.DateTime, default=_now)
    predictions = db.relationship("Prediction", backref="user", lazy="dynamic",
                                  cascade="all, delete-orphan")

    def set_password(self, password: str) -> None:
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)

    def regenerate_api_key(self) -> None:
        self.api_key = secrets.token_hex(20)

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    @property
    def is_staff(self) -> bool:
        return self.role in ("doctor", "admin")


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)
    module = db.Column(db.String(32), nullable=False, index=True)
    inputs_json = db.Column(db.Text, nullable=False, default="{}")
    label = db.Column(db.String(64), nullable=False)
    positive = db.Column(db.Boolean, nullable=False, default=False)
    probability = db.Column(db.Float, nullable=False)
    risk_level = db.Column(db.String(16), nullable=False)
    result_json = db.Column(db.Text, nullable=False, default="{}")
    model_name = db.Column(db.String(64))
    model_version = db.Column(db.String(32))
    source = db.Column(db.String(8), default="web")
    created_at = db.Column(db.DateTime, default=_now, index=True)

    @property
    def inputs(self) -> dict:
        return json.loads(self.inputs_json or "{}")

    @property
    def result(self) -> dict:
        return json.loads(self.result_json or "{}")

    @classmethod
    def from_result(cls, user, module: str, inputs: dict, result: dict, source="web"):
        # Privacy: raw mental-health text is never stored, only its length.
        safe_inputs = ({"text_length": len(inputs.get("text", ""))} if module == "mental_health"
                       else inputs)
        return cls(user_id=user.id, module=module, inputs_json=json.dumps(safe_inputs),
                   label=result["label"], positive=result["positive"],
                   probability=result["probability"], risk_level=result["risk_level"],
                   result_json=json.dumps({k: result[k] for k in
                                           ("factors", "details", "threshold")}),
                   model_name=result["model_name"], model_version=result["model_version"],
                   source=source)


class TrainingRun(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    module = db.Column(db.String(32), nullable=False)
    status = db.Column(db.String(16), nullable=False, default="running")  # running/success/failed
    promoted = db.Column(db.Boolean, default=False)
    log = db.Column(db.Text, default="")
    triggered_by = db.Column(db.Integer, db.ForeignKey("user.id"))
    started_at = db.Column(db.DateTime, default=_now)
    finished_at = db.Column(db.DateTime)
