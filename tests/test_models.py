"""Tests on the trained models (skipped if `python -m ml.train_all` has not been run)."""
import pytest

from ml.predict import Registry
from ml.schemas import validate
from tests.conftest import DIABETES, MENTAL, STROKE

registry = Registry().load()
CASES = [("diabetes", DIABETES), ("stroke", STROKE), ("mental_health", MENTAL)]


@pytest.mark.parametrize("module,data", CASES)
def test_prediction_shape(module, data):
    if registry.get(module) is None:
        pytest.skip("model not trained")
    values, _ = validate(module, data)
    r = registry.predict(module, values)
    assert set(r) >= {"label", "probability", "risk_level", "factors", "model_name"}
    assert 0.0 <= r["probability"] <= 1.0
    assert r["risk_level"] in ("Low", "Moderate", "High")


@pytest.mark.parametrize("module,data", CASES)
def test_reproducible(module, data):
    """Reliability NFR: the same input always gives the same output."""
    if registry.get(module) is None:
        pytest.skip("model not trained")
    values, _ = validate(module, data)
    assert registry.predict(module, values) == registry.predict(module, values)


def test_high_risk_profiles_are_flagged():
    if registry.get("diabetes") is None or registry.get("stroke") is None:
        pytest.skip("model not trained")
    d = registry.predict("diabetes", validate("diabetes", DIABETES)[0])
    s = registry.predict("stroke", validate("stroke", STROKE)[0])
    assert d["positive"] and s["positive"]


def test_low_risk_profile_not_flagged():
    if registry.get("diabetes") is None:
        pytest.skip("model not trained")
    healthy = dict(DIABETES, Pregnancies=0, Glucose=85, BMI=21.5, Age=24,
                   DiabetesPedigreeFunction=0.2)
    r = registry.predict("diabetes", validate("diabetes", healthy)[0])
    assert not r["positive"] and r["risk_level"] == "Low"


def test_mental_distress_detected_and_normal_text():
    if registry.get("mental_health") is None:
        pytest.skip("model not trained")
    r = registry.predict("mental_health", validate("mental_health", MENTAL)[0])
    assert r["details"]["distress"] is True
    normal = {"text": "Had a great lunch with friends today, then we watched a movie and "
                      "went for a walk in the park. Lovely weekend!", "consent": "on"}
    r2 = registry.predict("mental_health", validate("mental_health", normal)[0])
    assert r2["label"] == "Normal"
