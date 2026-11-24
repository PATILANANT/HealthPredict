"""Inference layer shared by the web app, the REST API and the tests.

Each predictor returns the same dictionary shape:
    {module, label, positive, probability, risk_level, factors, details, model_name, model_version}
"""
from __future__ import annotations

import threading
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml import safety
from ml.common.store import bundle_path, load_metrics
from ml.common.text import clean_text, set_known_words, stem_to_words
from ml.config import MODELS_DIR
from ml.schemas import MODULES, field_label, model_features, option_label

# Terms made only of these words carry no meaning on their own, so they are not shown.
FUNCTION_WORDS = frozenset({"not", "did", "do", "does", "my", "i", "me", "no"})

HELPLINE = safety.SELF_HARM_MSG
VIOLENT_LABEL = "Anger / violent thoughts"
# Chosen on the held-out test set: results with top probability < 0.50 or a gap < 0.15 to the
# runner-up (26% of texts) are correct only ~50% of the time, versus ~85% for all others.
UNCERTAIN_BELOW = 0.50
UNCERTAIN_MARGIN = 0.15


class TabularPredictor:
    def __init__(self, module: str, bundle: dict):
        self.module = module
        self.bundle = bundle
        self.pipeline = bundle["pipeline"]
        self.threshold = float(bundle["threshold"])
        self.features = bundle["features"]
        self.reference = bundle["reference"]
        self.spec = MODULES[module]

    def _frame(self, rows: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(rows, columns=self.features)
        return df.replace({None: np.nan}).infer_objects()

    def risk_level(self, p: float) -> str:
        if p >= self.threshold:
            return "High"
        if p >= self.threshold * 0.6:
            return "Moderate"
        return "Low"

    def predict(self, values: dict) -> dict:
        row = {f: values.get(f) for f in self.features}
        # What-if explanation: replace one feature at a time with a typical (median / most
        # common) value and measure how much the predicted risk changes.
        rows = [row]
        for f in self.features:
            alt = dict(row)
            alt[f] = self.reference[f]
            rows.append(alt)
        probs = self.pipeline.predict_proba(self._frame(rows))[:, 1]
        p = float(probs[0])
        factors = []
        for f, p_alt in zip(self.features, probs[1:]):
            delta = p - float(p_alt)
            if row[f] is None or abs(delta) < 0.01:
                continue
            factors.append({
                "feature": f,
                "label": field_label(self.module, f),
                "value": option_label(self.module, f, row[f]) if isinstance(row[f], str)
                or f in ("hypertension", "heart_disease") else row[f],
                "effect": round(delta, 4),
                "direction": "raises" if delta > 0 else "lowers",
            })
        factors.sort(key=lambda d: abs(d["effect"]), reverse=True)
        positive = p >= self.threshold
        return {
            "module": self.module,
            "label": self.spec["positive_label"] if positive else self.spec["negative_label"],
            "positive": bool(positive),
            "probability": round(p, 4),
            "threshold": self.threshold,
            "risk_level": self.risk_level(p),
            "factors": factors[:5],
            "details": {},
            "model_name": self.bundle["model_name"],
            "model_version": self.bundle["version"],
        }


class TextPredictor:
    def __init__(self, bundle: dict):
        self.module = "mental_health"
        self.bundle = bundle
        self.vec = bundle["vectorizer"]
        self.model = bundle["model"]
        self.explainer = bundle["explainer"]
        self.risk_map = bundle["risk_mapping"]
        self.vocab = np.array(self.vec.get_feature_names_out())
        set_known_words(bundle.get("known_words", {}))

    def predict(self, values: dict) -> dict:
        text = values["text"]
        X = self.vec.transform([clean_text(text)])
        proba = self.model.predict_proba(X)[0]
        classes = list(self.model.classes_)
        idx = int(np.argmax(proba))
        label = classes[idx]

        # Top words pushing towards the predicted class (from the linear explainer).
        words = []
        if X.nnz:
            ex_idx = list(self.explainer.classes_).index(label)
            coefs = self.explainer.coef_[ex_idx]
            cols = X.indices
            contrib = X.data * coefs[cols]
            order = np.argsort(-contrib)
            originals = stem_to_words(text)
            for j in order:
                if contrib[j] <= 0 or len(words) >= 6:
                    break
                parts = [originals.get(t, t) for t in self.vocab[cols[j]].split()]
                # "not" + "not nice" (negation word followed by its marked word) -> "not nice"
                if len(parts) == 2 and parts[0] == "not" and parts[1].startswith("not "):
                    parts = parts[1:]
                readable = " ".join(parts)
                if set(readable.split()) <= FUNCTION_WORDS or readable in words:
                    continue
                if len(parts) == 2 and (parts[-1] == "not" or all(p in words for p in parts)):
                    continue
                words.append(readable)

        ranking = sorted(zip(classes, proba), key=lambda kv: kv[1], reverse=True)
        model_label = label
        risk = self.risk_map.get(label, "Moderate")

        # Safety rules run on every text and can only raise the risk (see ml/safety.py).
        flags = safety.check(text)
        override = bool(flags["self_harm"] or flags["harm_to_others"])
        if flags["self_harm"]:
            label = "Suicidal"
        elif flags["harm_to_others"] and label == "Normal":
            label = VIOLENT_LABEL
        if override:
            risk = "High"
        factors = ([{"label": p, "feature": "safety"} for p in
                    flags["self_harm"] + flags["harm_to_others"]] +
                   [{"label": w, "feature": "word"} for w in words])
        return {
            "module": self.module,
            "label": label,
            "positive": label != "Normal",
            "probability": round(float(proba[idx]), 4),
            "threshold": None,
            "risk_level": risk,
            "factors": factors,
            "details": {
                "distress": label != "Normal",
                "class_probabilities": {c: round(float(p), 4) for c, p in ranking},
                "helpline": HELPLINE if label == "Suicidal" or
                dict(ranking).get("Suicidal", 0) >= 0.30 else None,
                "harm_alert": safety.HARM_OTHERS_MSG if flags["harm_to_others"] else None,
                "safety_override": override,
                # low confidence or a near-tie between the top two categories
                "uncertain": bool((not override) and (
                    ranking[0][1] < UNCERTAIN_BELOW or
                    ranking[0][1] - ranking[1][1] < UNCERTAIN_MARGIN)),
                "runner_up": ranking[1][0],
                "model_label": model_label,
                "text_length": len(text),
            },
            "model_name": self.bundle["model_name"],
            "model_version": self.bundle["version"],
        }


class Registry:
    """Loads every trained module once and keeps it in memory (fast predictions)."""

    def __init__(self, models_dir: Path | str = MODELS_DIR):
        self.models_dir = Path(models_dir)
        self.predictors: dict = {}
        self.errors: dict = {}
        self._lock = threading.Lock()

    def load(self) -> "Registry":
        predictors, errors = {}, {}
        for module in MODULES:
            path = bundle_path(module, self.models_dir)
            if not path.exists():
                errors[module] = "Model not trained yet. Run: python -m ml.train_all"
                continue
            try:
                bundle = joblib.load(path)
                predictors[module] = (TextPredictor(bundle) if module == "mental_health"
                                      else TabularPredictor(module, bundle))
            except Exception as exc:  # noqa: BLE001 - surface any load problem in the UI
                errors[module] = f"Could not load model: {exc}"
        with self._lock:
            self.predictors, self.errors = predictors, errors
        return self

    def get(self, module: str):
        return self.predictors.get(module)

    def metrics(self, module: str) -> dict | None:
        return load_metrics(module, self.models_dir)

    def predict(self, module: str, values: dict) -> dict:
        predictor = self.get(module)
        if predictor is None:
            raise RuntimeError(self.errors.get(module, "Model unavailable"))
        return predictor.predict({k: values.get(k) for k in model_features(module)})
