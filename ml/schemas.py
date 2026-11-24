"""Input schema for every prediction module.

The same definitions drive (1) the HTML forms, (2) server-side validation for the
web app and REST API (FR-14) and (3) the column order fed to the trained models.
"""
from __future__ import annotations

import math
import re

YES_NO = [(0, "No"), (1, "Yes")]

MODULES: dict[str, dict] = {
    "mental_health": {
        "title": "Mental Health",
        "icon": "bi-chat-heart",
        "summary": "Screens a piece of social-media style text for signs of depression, anxiety, "
                   "stress and other conditions.",
        "dataset": "Sentiment Analysis for Mental Health (Kaggle, ~53k statements)",
        "fields": [
            {"name": "text", "label": "Your text / post", "type": "text", "min_length": 15,
             "max_length": 5000,
             "help": "Paste a post, comment or a few sentences about how you have been feeling (English only)."},
            {"name": "consent", "label": "I agree that this text is analysed for screening only. "
                                         "The text itself is not stored.",
             "type": "consent"},
        ],
    },
    "stroke": {
        "title": "Brain Stroke",
        "icon": "bi-activity",
        "summary": "Estimates the risk of brain stroke from demographic details and medical history.",
        "dataset": "Stroke Prediction Dataset (Kaggle, 5,110 patients)",
        "positive_label": "High Stroke Risk",
        "negative_label": "Low Stroke Risk",
        "fields": [
            {"name": "gender", "label": "Gender", "type": "select",
             "options": [("Male", "Male"), ("Female", "Female"), ("Other", "Other")]},
            {"name": "age", "label": "Age (years)", "type": "float", "min": 1, "max": 100, "step": 1},
            {"name": "hypertension", "label": "Hypertension (high blood pressure)", "type": "select",
             "options": YES_NO},
            {"name": "heart_disease", "label": "Heart disease", "type": "select", "options": YES_NO},
            {"name": "ever_married", "label": "Ever married", "type": "select",
             "options": [("Yes", "Yes"), ("No", "No")]},
            {"name": "work_type", "label": "Work type", "type": "select",
             "options": [("Private", "Private"), ("Self-employed", "Self-employed"),
                         ("Govt_job", "Government job"), ("children", "Child"),
                         ("Never_worked", "Never worked")]},
            {"name": "Residence_type", "label": "Residence type", "type": "select",
             "options": [("Urban", "Urban"), ("Rural", "Rural")]},
            {"name": "avg_glucose_level", "label": "Average glucose level (mg/dL)", "type": "float",
             "min": 50, "max": 300, "step": 0.1},
            {"name": "bmi", "label": "BMI", "type": "float", "min": 10, "max": 70, "step": 0.1,
             "required": False, "help": "Leave blank if unknown."},
            {"name": "smoking_status", "label": "Smoking status", "type": "select",
             "options": [("never smoked", "Never smoked"), ("formerly smoked", "Formerly smoked"),
                         ("smokes", "Smokes"), ("Unknown", "Prefer not to say / unknown")]},
        ],
    },
    "diabetes": {
        "title": "Diabetes",
        "icon": "bi-droplet-half",
        "summary": "Estimates the likelihood of diabetes from routine clinical measurements.",
        "dataset": "Pima Indians Diabetes Database (768 records)",
        "positive_label": "Diabetic",
        "negative_label": "Not Diabetic",
        "fields": [
            {"name": "Pregnancies", "label": "Pregnancies", "type": "int", "min": 0, "max": 20,
             "step": 1, "help": "Number of times pregnant (0 if none)."},
            {"name": "Glucose", "label": "Glucose (mg/dL, 2-hour OGTT)", "type": "float",
             "min": 40, "max": 300, "step": 1},
            {"name": "BloodPressure", "label": "Diastolic blood pressure (mm Hg)", "type": "float",
             "min": 30, "max": 150, "step": 1},
            {"name": "SkinThickness", "label": "Triceps skin-fold thickness (mm)", "type": "float",
             "min": 5, "max": 100, "step": 1, "required": False, "help": "Leave blank if unknown."},
            {"name": "Insulin", "label": "2-hour serum insulin (mu U/ml)", "type": "float",
             "min": 10, "max": 900, "step": 1, "required": False, "help": "Leave blank if unknown."},
            {"name": "BMI", "label": "BMI", "type": "float", "min": 10, "max": 70, "step": 0.1},
            {"name": "DiabetesPedigreeFunction", "label": "Diabetes pedigree function",
             "type": "float", "min": 0.05, "max": 2.5, "step": 0.001, "required": False,
             "help": "Family-history score. Leave blank if unknown."},
            {"name": "Age", "label": "Age (years)", "type": "int", "min": 18, "max": 100, "step": 1},
        ],
    },
}


# ---- presentation metadata (units, form sections, normal ranges, colours) ------------------
_UI = {
    "mental_health": {"color": "violet", "tagline": "Text-based screening",
                      "fields": {"text": {"group": "Tell us how you feel"},
                                 "consent": {"group": "Tell us how you feel"}}},
    "stroke": {"color": "rose", "tagline": "Clinical risk estimate", "fields": {
        "gender": {"group": "About you"}, "age": {"group": "About you", "unit": "years"},
        "ever_married": {"group": "About you"}, "work_type": {"group": "About you"},
        "Residence_type": {"group": "About you"},
        "hypertension": {"group": "Medical history & lifestyle"},
        "heart_disease": {"group": "Medical history & lifestyle"},
        "smoking_status": {"group": "Medical history & lifestyle"},
        "avg_glucose_level": {"group": "Measurements", "unit": "mg/dL", "normal": (70, 140)},
        "bmi": {"group": "Measurements", "unit": "kg/m²", "normal": (18.5, 24.9), "bmi_calc": True},
    }},
    "diabetes": {"color": "sky", "tagline": "Clinical risk estimate", "fields": {
        "Age": {"group": "About you", "unit": "years"},
        "Pregnancies": {"group": "About you"},
        "DiabetesPedigreeFunction": {"group": "About you"},
        "Glucose": {"group": "Blood tests", "unit": "mg/dL", "normal": (70, 140)},
        "Insulin": {"group": "Blood tests", "unit": "µU/ml", "normal": (16, 166)},
        "BloodPressure": {"group": "Body measurements", "unit": "mm Hg", "normal": (60, 80)},
        "BMI": {"group": "Body measurements", "unit": "kg/m²", "normal": (18.5, 24.9), "bmi_calc": True},
        "SkinThickness": {"group": "Body measurements", "unit": "mm"},
    }},
}
for _m, _meta in _UI.items():
    MODULES[_m]["color"] = _meta["color"]
    MODULES[_m]["tagline"] = _meta["tagline"]
    for _f in MODULES[_m]["fields"]:
        _f.update(_meta["fields"].get(_f["name"], {}))
        if _f.get("unit"):  # the unit is shown next to the input, so drop "(mg/dL)" etc.
            _f["label"] = re.sub(r"\s*\([^)]*\)$", "", _f["label"])
MODULES["diabetes"]["fields"][1]["help"] = "From a 2-hour oral glucose tolerance test (OGTT)."


def grouped_fields(module: str) -> list[tuple[str, list[dict]]]:
    """Fields grouped into form sections, keeping the first-seen order of each section."""
    groups: dict[str, list[dict]] = {}
    for f in MODULES[module]["fields"]:
        groups.setdefault(f.get("group", "Details"), []).append(f)
    return list(groups.items())


def range_status(field: dict, value) -> str | None:
    """'low' / 'normal' / 'high' compared with the field's normal range (None if no range)."""
    rng = field.get("normal")
    if rng is None or value is None:
        return None
    return "low" if value < rng[0] else ("high" if value > rng[1] else "normal")


def model_features(module: str) -> list[str]:
    """Columns that are passed to the model (consent boxes are excluded)."""
    return [f["name"] for f in MODULES[module]["fields"] if f["type"] != "consent"]


def option_label(module: str, field: str, value) -> str:
    for f in MODULES[module]["fields"]:
        if f["name"] == field and f["type"] == "select":
            for v, label in f["options"]:
                if str(v) == str(value):
                    return label
    return "-" if value is None else str(value)


def field_label(module: str, field: str) -> str:
    for f in MODULES[module]["fields"]:
        if f["name"] == field:
            return f["label"]
    return field


def validate(module: str, data) -> tuple[dict, dict]:
    """Validate raw form/JSON input. Returns (clean_values, errors)."""
    if module not in MODULES:
        raise KeyError(module)
    values: dict = {}
    errors: dict = {}
    for f in MODULES[module]["fields"]:
        name, kind, label = f["name"], f["type"], f["label"]
        raw = data.get(name) if hasattr(data, "get") else None
        if isinstance(raw, str):
            raw = raw.strip()

        if kind == "consent":
            if str(raw).lower() not in ("on", "true", "1", "yes"):
                errors[name] = "Please give consent before the text is analysed."
            continue

        if raw is None or raw == "":
            if f.get("required", True):
                errors[name] = f"{label} is required."
            else:
                values[name] = None
            continue

        if kind in ("int", "float"):
            try:
                v = float(raw)
            except (TypeError, ValueError):
                errors[name] = f"{label} must be a number."
                continue
            if not math.isfinite(v):
                errors[name] = f"{label} must be a number."
                continue
            if kind == "int" and v != int(v):
                errors[name] = f"{label} must be a whole number."
                continue
            if v < f["min"] or v > f["max"]:
                errors[name] = f"{label} must be between {f['min']} and {f['max']}."
                continue
            values[name] = int(v) if kind == "int" else v
        elif kind == "select":
            allowed = {str(v): v for v, _ in f["options"]}
            if str(raw) not in allowed:
                errors[name] = f"Choose a valid option for {label}."
            else:
                values[name] = allowed[str(raw)]
        elif kind == "text":
            text = str(raw)
            if len(text) < f.get("min_length", 1):
                errors[name] = f"Please enter at least {f['min_length']} characters."
            elif len(text) > f.get("max_length", 10_000):
                errors[name] = f"Please keep the text under {f['max_length']} characters."
            elif sum(ch.isalpha() for ch in text) < 10:
                errors[name] = "The text must contain words (English)."
            else:
                values[name] = text
    return values, errors
