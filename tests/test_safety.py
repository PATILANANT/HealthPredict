import pytest

from ml.predict import Registry
from ml.safety import check

registry = Registry().load()

HARM_OTHERS = [
    "i loved my gf soooo muchh... but my girlfrriend left me for another guy... I really loved her "
    "so  muchh.. I wont spare that guyy.. he will diserve worsttt.. i want to kill himm",
    "My boss is so annoying, I want to punch him",
    "She deserves the worst after what she did, I will make her pay",
    "im gonna hurt that guy tomorrow",
]
SELF_HARM = [
    "I do not want to live anymore",
    "thinking about ending my life",
    "I am gonna kill myself",
    "i want to dieee",
]
HARMLESS = [
    "This traffic is killing me, worst day",
    "I will kill this exam tomorrow, so prepared!",
    "I killed it at the presentation today",
    "Had a really nice day, played cricket and had dinner",
    "My sister and I watched a murder mystery movie",
]


@pytest.mark.parametrize("text", HARM_OTHERS)
def test_harm_to_others_detected(text):
    assert check(text)["harm_to_others"]


@pytest.mark.parametrize("text", SELF_HARM)
def test_self_harm_detected(text):
    assert check(text)["self_harm"]


@pytest.mark.parametrize("text", HARMLESS)
def test_slang_not_flagged(text):
    flags = check(text)
    assert not flags["self_harm"] and not flags["harm_to_others"]


def test_phrase_shown_in_users_spelling():
    assert "kill himm" in check(HARM_OTHERS[0])["harm_to_others"][0]


def test_override_raises_risk():
    if registry.get("mental_health") is None:
        pytest.skip("model not trained")
    r = registry.predict("mental_health", {"text": HARM_OTHERS[0]})
    assert r["risk_level"] == "High" and r["label"] != "Normal"
    assert r["details"]["harm_alert"] and r["details"]["safety_override"]
    assert r["factors"][0]["feature"] == "safety"
    r2 = registry.predict("mental_health", {"text": SELF_HARM[2] + " i have no hope left"})
    assert r2["label"] == "Suicidal" and r2["details"]["helpline"]


@pytest.mark.parametrize("text", ["i wanna dieee plzz", "I wish I was dead", "gonna end it tonight"])
def test_more_self_harm_forms(text):
    assert check(text)["self_harm"]


@pytest.mark.parametrize("text", ["die hard fan of dhoni", "I wanna diet this month",
                                  "I died laughing", "my feet are dying"])
def test_more_harmless_forms(text):
    flags = check(text)
    assert not flags["self_harm"] and not flags["harm_to_others"]
