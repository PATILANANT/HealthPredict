import pytest

from app import create_app
from app.extensions import db
from app.models import User


@pytest.fixture()
def app(tmp_path):
    app = create_app({"TESTING": True, "WTF_CSRF_ENABLED": False,
                      "SQLALCHEMY_DATABASE_URI": f"sqlite:///{tmp_path / 'test.db'}"})
    yield app
    with app.app_context():
        db.session.remove()
        db.engine.dispose()


@pytest.fixture()
def client(app):
    return app.test_client()


def make_user(app, email="user@example.com", role="user", password="password123"):
    with app.app_context():
        u = User(name=email.split("@")[0].title(), email=email, role=role)
        u.set_password(password)
        db.session.add(u)
        db.session.commit()
        return u.id, u.api_key


@pytest.fixture()
def user(app):
    return make_user(app)


@pytest.fixture()
def logged_in(client, user):
    client.post("/login", data={"email": "user@example.com", "password": "password123"})
    return client


DIABETES = {"Pregnancies": 2, "Glucose": 165, "BloodPressure": 80, "SkinThickness": "",
            "Insulin": "", "BMI": 34.5, "DiabetesPedigreeFunction": 0.6, "Age": 50}
STROKE = {"gender": "Male", "age": 72, "hypertension": 1, "heart_disease": 1, "ever_married": "Yes",
          "work_type": "Private", "Residence_type": "Urban", "avg_glucose_level": 210.5,
          "bmi": 31.0, "smoking_status": "formerly smoked"}
MENTAL = {"text": "I can't sleep anymore, everything feels hopeless and I don't see the point "
                  "of anything. I just want to be alone all the time.", "consent": "on"}
