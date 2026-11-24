"""Behaviour tests on realistic / tricky inputs.

`MUST` cases have to pass (safety-critical or clear-cut). `KNOWN_LIMITS` document where the
model is still wrong; they are marked xfail so the suite stays green but the gaps stay visible.
"""
import pytest

from ml.predict import Registry

r = Registry().load()
pytestmark = pytest.mark.skipif(r.get("mental_health") is None, reason="model not trained")

DISTRESS = {"Depression", "Suicidal", "Anxiety", "Stress", "Bipolar", "Personality disorder",
            "Anger / violent thoughts"}

MUST = [
    ("Going to the gym after work, then pizza with my roommates. Pretty chill week.", {"Normal"}),
    ("Just got promoted at work!! So excited, celebrating with family tonight", {"Normal"}),
    ("Watching the India vs Australia match, what a game!", {"Normal"}),
    ("I am not sad at all, actually I feel great today", {"Normal"}),
    ("aaj ka din bahut accha tha, friends ke saath movie dekhi", {"Normal"}),
    ("This assignment is killing me lol but almost done", {"Normal", "Stress"}),
    ("i feel so emptyyy and numb, nothing matters anymoreee", {"Depression", "Suicidal"}),
    ("I am not happy, I don't enjoy anything, I feel like a burden on my family", {"Depression", "Suicidal"}),
    ("lost my job, my wife left, I cry every night and feel worthless", {"Depression", "Suicidal"}),
    ("I have pills saved up and tonight might be the night", {"Suicidal", "Depression"}),
    ("there is no point living like this, I want to end it all", {"Suicidal"}),
    ("i wanna dieee plzz", {"Suicidal"}),
    ("I keep thinking about jumping off the terrace of my building", {"Suicidal"}),
    ("I keep overthinking everything and can't sleep because of constant worry about my health", {"Anxiety", "Stress"}),
    ("Board exams in 5 days, parents pressuring me, I'm so stressed I get headaches", {"Stress", "Anxiety"}),
    ("I was manic for a week, didn't sleep, now I'm in a deep depressive episode again", {"Bipolar"}),
    ("He cheated me of my money, I am going to beat him up when I find him", DISTRESS),
    ("my stepdad hits my mom, one day I will kill him", DISTRESS),
    ("I want to stab that guy who bullied me", DISTRESS),
]

KNOWN_LIMITS = [
    ("Everyone says it gets better but it never does. I'm tired of pretending to be okay.", {"Depression", "Suicidal"}),
    ("Deadlines everywhere, office pressure is too much, I'm burnt out", {"Stress", "Anxiety"}),
    ("The comedian killed it last night, I died laughing", {"Normal"}),
    ("My feet are dying after that 10k run", {"Normal"}),
]


@pytest.mark.parametrize("text,allowed", MUST)
def test_expected_label(text, allowed):
    out = r.predict("mental_health", {"text": text})
    assert out["label"] in allowed, (out["label"], out["probability"])


@pytest.mark.parametrize("text,allowed", KNOWN_LIMITS)
@pytest.mark.xfail(reason="known model limitation (subtle distress / figurative language)", strict=False)
def test_known_limitations(text, allowed):
    assert r.predict("mental_health", {"text": text})["label"] in allowed


def test_self_harm_always_high_risk():
    for text in ("i wanna dieee plzz", "I wish I was dead", "gonna end it tonight"):
        out = r.predict("mental_health", {"text": text})
        assert out["risk_level"] == "High" and out["details"]["helpline"], text


@pytest.mark.parametrize("text", [
    "I split people into all good or all bad and my relationships always end in explosive "
    "fights. My psychiatrist says BPD.",
    "Heart pounding, sweaty palms, feel like I'm going to faint before my interview",
])
def test_uncertain_flag_for_ambiguous_text(text):
    assert r.predict("mental_health", {"text": text})["details"]["uncertain"] is True


def test_confident_text_not_uncertain():
    out = r.predict("mental_health", {"text": "Watching the India vs Australia match, what a game!"})
    assert out["details"]["uncertain"] is False


# ---- clinical sanity: risk must move in the medically expected direction ----
BASE_D = dict(Pregnancies=1, Glucose=110, BloodPressure=72, SkinThickness=None, Insulin=None,
              BMI=27.0, DiabetesPedigreeFunction=0.4, Age=35)
BASE_S = dict(gender="Female", age=50, hypertension=0, heart_disease=0, ever_married="Yes",
              work_type="Private", Residence_type="Urban", avg_glucose_level=100.0, bmi=27.0,
              smoking_status="never smoked")


def prob(module, **kw):
    return r.predict(module, dict(BASE_D if module == "diabetes" else BASE_S, **kw))["probability"]


def test_diabetes_risk_rises_with_glucose_and_bmi():
    g = [prob("diabetes", Glucose=v) for v in (80, 110, 140, 170, 200)]
    assert all(b >= a - 0.02 for a, b in zip(g, g[1:])) and g[-1] > g[0] + 0.2
    assert prob("diabetes", BMI=45) > prob("diabetes", BMI=20)


def test_stroke_risk_rises_with_age_and_conditions():
    a = [prob("stroke", age=v) for v in (20, 40, 60, 80)]
    assert all(y >= x - 0.02 for x, y in zip(a, a[1:])) and a[-1] > a[0] + 0.3
    assert prob("stroke", hypertension=1) > prob("stroke")
    assert prob("stroke", heart_disease=1) > prob("stroke")
    assert prob("stroke", avg_glucose_level=250.0) > prob("stroke")


def test_healthy_profiles_low():
    assert prob("stroke", age=8, work_type="children", ever_married="No") < 0.2
    assert prob("diabetes", Pregnancies=0, Glucose=85, BMI=21, Age=22, DiabetesPedigreeFunction=0.2) < 0.25


@pytest.mark.parametrize("module,kw", [
    ("diabetes", dict(Glucose=40, BMI=10, Age=18, BloodPressure=30)),
    ("diabetes", dict(Glucose=300, BMI=70, Age=100, BloodPressure=150)),
    ("stroke", dict(age=1, avg_glucose_level=50, bmi=10)),
    ("stroke", dict(age=100, avg_glucose_level=300, bmi=None)),
])
def test_extreme_valid_inputs(module, kw):
    assert 0.0 <= prob(module, **kw) <= 1.0
