import pytest

from app.extensions import registry
from app.models import Prediction
from tests.conftest import DIABETES, MENTAL, STROKE, make_user


def test_public_pages(client):
    assert client.get("/").status_code == 200
    assert client.get("/about").status_code == 200
    assert client.get("/api/v1/health").status_code == 200


def test_register_login_logout(client):
    r = client.post("/register", data={"name": "Asha", "email": "asha@example.com",
                                       "password": "secret123", "confirm": "secret123"})
    assert r.status_code == 302
    assert client.get("/dashboard").status_code == 200
    client.post("/logout")
    assert client.get("/dashboard").status_code == 302  # redirected to login


def test_register_rejects_bad_input(client):
    r = client.post("/register", data={"name": "A", "email": "bad", "password": "1",
                                       "confirm": "2"})
    assert r.status_code == 200 and b"valid email" in r.data


def test_predict_requires_login(client):
    assert client.get("/predict/diabetes").status_code == 302


@pytest.mark.parametrize("module,data", [("diabetes", DIABETES), ("stroke", STROKE),
                                         ("mental_health", MENTAL)])
def test_web_prediction_flow(app, logged_in, module, data):
    if registry.get(module) is None:
        pytest.skip("model not trained")
    r = logged_in.post(f"/predict/{module}", data=data)
    assert r.status_code == 302 and "/result/" in r.headers["Location"]
    page = logged_in.get(r.headers["Location"])
    assert page.status_code == 200 and b"screening result" in page.data
    with app.app_context():
        p = Prediction.query.one()
        assert p.module == module
        if module == "mental_health":   # privacy: raw text never stored
            assert "text" not in p.inputs and "hopeless" not in p.inputs_json
    pdf = logged_in.get(f"/report/pdf/{p.id}")
    assert pdf.status_code == 200 and pdf.data[:4] == b"%PDF"


def test_validation_errors_shown(logged_in):
    r = logged_in.post("/predict/diabetes", data=dict(DIABETES, Glucose="abc"))
    assert r.status_code == 200 and b"must be a number" in r.data


def test_other_users_result_is_forbidden(app, client, logged_in):
    if registry.get("diabetes") is None:
        pytest.skip("model not trained")
    r = logged_in.post("/predict/diabetes", data=DIABETES)
    url = r.headers["Location"]
    logged_in.post("/logout")
    make_user(app, "other@example.com")
    client.post("/login", data={"email": "other@example.com", "password": "password123"})
    assert client.get(url).status_code == 403


def test_role_protection(app, client):
    make_user(app, "doc@example.com", role="doctor")
    client.post("/login", data={"email": "doc@example.com", "password": "password123"})
    assert client.get("/patients/").status_code == 200
    assert client.get("/admin/").status_code == 403


def test_admin_pages(app, client):
    make_user(app, "admin@example.com", role="admin")
    client.post("/login", data={"email": "admin@example.com", "password": "password123"})
    for url in ("/admin/", "/admin/users", "/admin/models", "/patients/"):
        assert client.get(url).status_code == 200, url
    for module in ("diabetes", "stroke", "mental_health"):
        assert client.get(f"/metrics/{module}").status_code == 200


def test_api(app, client, user):
    _, key = user
    assert client.post("/api/v1/predict/diabetes", json=DIABETES).status_code == 401
    r = client.post("/api/v1/predict/diabetes", json=dict(DIABETES, Glucose=1000),
                    headers={"X-API-Key": key})
    assert r.status_code == 422 and "Glucose" in r.get_json()["fields"]
    if registry.get("diabetes") is None:
        pytest.skip("model not trained")
    r = client.post("/api/v1/predict/diabetes", json=DIABETES, headers={"X-API-Key": key})
    body = r.get_json()
    assert r.status_code == 200 and body["label"] in ("Diabetic", "Not Diabetic")
    hist = client.get("/api/v1/predictions", headers={"X-API-Key": key}).get_json()
    assert hist[0]["id"] == body["prediction_id"]
    assert "diabetes" in client.get("/api/v1/models").get_json()
