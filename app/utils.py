from functools import wraps

from flask import abort
from flask_login import current_user, login_required


def roles_required(*roles):
    def decorator(view):
        @wraps(view)
        @login_required
        def wrapped(*args, **kwargs):
            if current_user.role not in roles:
                abort(403)
            return view(*args, **kwargs)
        return wrapped
    return decorator


def latest_per_module(user):
    """The most recent prediction of each module for one user (for the combined report)."""
    from ml.schemas import MODULES
    from .models import Prediction
    out = {}
    for module in MODULES:
        p = (user.predictions.filter(Prediction.module == module)
             .order_by(Prediction.created_at.desc()).first())
        if p:
            out[module] = p
    return out
