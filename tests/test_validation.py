from ml.common.text import clean_text
from ml.schemas import validate
from tests.conftest import DIABETES, MENTAL, STROKE


def test_valid_inputs_pass():
    for module, data in (("diabetes", DIABETES), ("stroke", STROKE), ("mental_health", MENTAL)):
        values, errors = validate(module, data)
        assert errors == {}, (module, errors)


def test_optional_blank_becomes_none():
    values, _ = validate("diabetes", DIABETES)
    assert values["Insulin"] is None and values["SkinThickness"] is None


def test_out_of_range_and_missing_are_rejected():
    bad = dict(DIABETES, Glucose=900, BMI="")
    _, errors = validate("diabetes", bad)
    assert "Glucose" in errors and "BMI" in errors


def test_non_numeric_and_bad_option_rejected():
    _, errors = validate("stroke", dict(STROKE, age="old", work_type="Astronaut"))
    assert "age" in errors and "work_type" in errors


def test_whole_number_required():
    _, errors = validate("diabetes", dict(DIABETES, Pregnancies=1.5))
    assert "Pregnancies" in errors


def test_mental_requires_consent_and_text():
    _, errors = validate("mental_health", {"text": "hi"})
    assert "consent" in errors and "text" in errors


def test_text_cleaning_keeps_negations():
    cleaned = clean_text("I can't sleep and I won't eat https://example.com @friend")
    assert "not" in cleaned.split()
    assert "http" not in cleaned and "friend" not in cleaned
    assert cleaned.split()[0] == "i"


def test_elongated_words_normalised():
    from ml.common.text import fix_elongation, set_known_words
    set_known_words({"die": 100, "empty": 50, "so": 900, "him": 300, "good": 500, "sleep": 200})
    assert [fix_elongation(w) for w in ("dieee", "emptyyy", "soooo", "himm", "gooood")] == \
        ["die", "empty", "so", "him", "good"]
    assert fix_elongation("sleep") == "sleep" and fix_elongation("good") == "good"
    set_known_words({})
