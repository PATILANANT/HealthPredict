import click

from .extensions import db
from .models import ROLES, User


DEMO_ACCOUNTS = [
    ("Admin", "admin@healthpredict.local", "admin12345", "admin"),
    ("Dr. Demo", "doctor@healthpredict.local", "doctor12345", "doctor"),
    ("Ganesh Sonawane", "ganesh@example.com", "ganesh12345", "user"),
]


def register_cli(app):
    @app.cli.command("seed-demo")
    def seed_demo():
        """Create the demo accounts if they do not exist yet (never changes existing ones)."""
        for name, email, password, role in DEMO_ACCOUNTS:
            if User.query.filter_by(email=email).first():
                continue
            user = User(name=name, email=email, role=role)
            user.set_password(password)
            db.session.add(user)
            click.echo(f"Created demo {role}: {email} / {password}")
        db.session.commit()

    @app.cli.command("create-user")
    @click.option("--name", required=True)
    @click.option("--email", required=True)
    @click.option("--password", required=True)
    @click.option("--role", type=click.Choice(ROLES), default="user")
    def create_user(name, email, password, role):
        """Create a user, e.g. the first administrator."""
        email = email.strip().lower()
        user = User.query.filter_by(email=email).first()
        if user:
            user.role = role
            user.set_password(password)
            click.echo(f"Updated existing user {email} (role={role})")
        else:
            user = User(name=name, email=email, role=role)
            user.set_password(password)
            db.session.add(user)
            click.echo(f"Created {role} {email}")
        db.session.commit()
