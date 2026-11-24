"""Registration, login, logout and profile (FR-01)."""
import re

from flask import Blueprint, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required, login_user, logout_user

from .extensions import db
from .models import User

bp = Blueprint("auth", __name__)
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


@bp.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("main.dashboard"))
    errors, form = {}, {}
    if request.method == "POST":
        form = {k: request.form.get(k, "").strip() for k in ("name", "email")}
        password = request.form.get("password", "")
        confirm = request.form.get("confirm", "")
        form["email"] = form["email"].lower()
        if len(form["name"]) < 2:
            errors["name"] = "Please enter your name."
        if not EMAIL_RE.match(form["email"]):
            errors["email"] = "Please enter a valid email address."
        elif User.query.filter_by(email=form["email"]).first():
            errors["email"] = "An account with this email already exists."
        if len(password) < 8:
            errors["password"] = "Password must be at least 8 characters."
        elif password != confirm:
            errors["confirm"] = "Passwords do not match."
        if not errors:
            user = User(name=form["name"], email=form["email"], role="user")
            user.set_password(password)
            db.session.add(user)
            db.session.commit()
            login_user(user)
            flash("Welcome! Your account has been created.", "success")
            return redirect(url_for("main.dashboard"))
    return render_template("auth/register.html", errors=errors, form=form)


@bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("main.dashboard"))
    error = None
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        user = User.query.filter_by(email=email).first()
        if user and user.check_password(request.form.get("password", "")):
            login_user(user, remember=bool(request.form.get("remember")))
            nxt = request.args.get("next", "")
            if not nxt.startswith("/") or nxt.startswith("//"):
                nxt = url_for("main.dashboard")
            return redirect(nxt)
        error = "Invalid email or password."
    return render_template("auth/login.html", error=error)


@bp.route("/logout", methods=["POST"])
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for("main.index"))


@bp.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    if request.method == "POST" and request.form.get("action") == "regenerate":
        current_user.regenerate_api_key()
        db.session.commit()
        flash("A new API key has been generated.", "success")
        return redirect(url_for("auth.profile"))
    return render_template("auth/profile.html")
