"""Train and benchmark the Mental Health module (social-media text -> 7 categories).

Pipeline: clean + stem text -> TF-IDF (word 1-2 grams) -> classifier.
Split: 70% train / 10% validation (model selection) / 20% test (final report only).

Usage:  python -m ml.mental_health.train [--force] [--data path.csv] [--fast]
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

from ml.config import DATA_FILES, RANDOM_STATE, REQUIRED_COLUMNS, TEST_SIZE
from ml.common.evaluation import _r, multiclass_metrics
from ml.common.store import save_bundle
from ml.common.text import build_known_words, clean_many, set_known_words

RISK_LEVEL = {
    "Normal": "Low",
    "Stress": "Moderate",
    "Anxiety": "Moderate",
    "Bipolar": "Moderate",
    "Personality disorder": "Moderate",
    "Depression": "High",
    "Suicidal": "High",
}


def load(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(REQUIRED_COLUMNS["mental_health"]) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    df = df[["statement", "status"]].dropna()
    df["statement"] = df["statement"].astype(str).str.strip()
    df = df[df["statement"].str.len() > 0]
    # identical statements with different labels are ambiguous -> keep the first one only
    return df.drop_duplicates(subset="statement").reset_index(drop=True)


def candidates(rs: int, fast: bool) -> dict:
    lr = lambda: LogisticRegression(C=5.0, max_iter=2000, class_weight="balanced")
    svm = lambda: CalibratedClassifierCV(LinearSVC(C=0.3, class_weight="balanced"), cv=3)
    nb = lambda: ComplementNB(alpha=0.3)
    models = {
        "Naive Bayes": nb(),
        "Logistic Regression": lr(),
        "Linear SVM": svm(),
        "SGD (Modified Huber)": SGDClassifier(loss="modified_huber", alpha=2e-5,
                                              class_weight="balanced", max_iter=50,
                                              random_state=rs),
        "K-Nearest Neighbours": KNeighborsClassifier(n_neighbors=25, metric="cosine",
                                                     weights="distance"),
        "Voting Ensemble": VotingClassifier([("lr", lr()), ("svm", svm()), ("nb", nb())],
                                            voting="soft", weights=[2, 2, 1]),
    }
    if not fast:
        models["Random Forest"] = RandomForestClassifier(
            n_estimators=200, n_jobs=-1, class_weight="balanced_subsample", random_state=rs)
    return models


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA_FILES["mental_health"]))
    ap.add_argument("--force", action="store_true", help="promote even if not better")
    ap.add_argument("--fast", action="store_true", help="skip the slow Random Forest")
    args = ap.parse_args(argv)
    log = print

    t0 = time.time()
    df = load(args.data)
    labels = sorted(df["status"].unique().tolist())
    log(f"[mental_health] {len(df)} statements, classes: {df['status'].value_counts().to_dict()}")
    known_words = build_known_words(df["statement"])
    set_known_words(known_words)
    df["clean"] = clean_many(df["statement"])
    df = df[df["clean"].str.len() > 0]
    log(f"[mental_health] cleaned in {time.time() - t0:.1f}s")

    trval, test = train_test_split(df, test_size=TEST_SIZE, stratify=df["status"],
                                   random_state=RANDOM_STATE)
    train, val = train_test_split(trval, test_size=0.125, stratify=trval["status"],
                                  random_state=RANDOM_STATE)

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, max_features=150_000,
                          sublinear_tf=True, dtype=np.float32)
    Xtr = vec.fit_transform(train["clean"])
    Xva, Xte = vec.transform(val["clean"]), vec.transform(test["clean"])
    ytr, yva, yte = train["status"].to_numpy(), val["status"].to_numpy(), test["status"].to_numpy()
    log(f"[mental_health] train={Xtr.shape[0]} val={Xva.shape[0]} test={Xte.shape[0]} "
        f"features={Xtr.shape[1]}")

    results, fitted = {}, {}
    for name, model in candidates(RANDOM_STATE, args.fast).items():
        t1 = time.time()
        model.fit(Xtr, ytr)
        results[name] = {
            "val": multiclass_metrics(yva, model.predict(Xva), labels),
            "test": multiclass_metrics(yte, model.predict(Xte), labels),
            "seconds": round(time.time() - t1, 1),
        }
        fitted[name] = model
        v, t = results[name]["val"], results[name]["test"]
        log(f"  {name:<22} val macro-F1={v['f1']:.3f} | test acc={t['accuracy']:.3f} "
            f"macro-P={t['precision']:.3f} macro-R={t['recall']:.3f} macro-F1={t['f1']:.3f} "
            f"({results[name]['seconds']}s)")

    best = max(results, key=lambda n: results[n]["val"]["f1"])
    log(f"[mental_health] best model: {best} (val macro-F1={results[best]['val']['f1']:.4f})")

    # Logistic Regression is kept as the "explainer": its coefficients show which words pushed
    # the prediction towards the predicted category.
    explainer = fitted["Logistic Regression"]
    # Binary Distress / No-Distress view (FR-09): everything except Normal counts as distress.
    yb_true = (yte != "Normal").astype(int)
    yb_pred = (fitted[best].predict(Xte) != "Normal").astype(int)
    distress = {"accuracy": _r(accuracy_score(yb_true, yb_pred)),
                "precision": _r(precision_score(yb_true, yb_pred)),
                "recall": _r(recall_score(yb_true, yb_pred)),
                "f1": _r(f1_score(yb_true, yb_pred))}
    log(f"[mental_health] distress vs normal on test: {distress}")

    version = datetime.now().strftime("%Y.%m.%d-%H%M%S")
    report = {
        "module": "mental_health", "task": "multiclass", "version": version,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "best_model": best, "selection_metric": "val_macro_f1",
        "selection_score": results[best]["val"]["f1"],
        "n_train": int(Xtr.shape[0]), "n_val": int(Xva.shape[0]), "n_test": int(Xte.shape[0]),
        "n_features": int(Xtr.shape[1]), "labels": labels,
        "dataset": {"file": str(args.data), "rows": int(len(df)),
                    "class_counts": df["status"].value_counts().to_dict()},
        "distress_binary_test": distress, "risk_mapping": RISK_LEVEL, "models": results,
    }
    bundle = {
        "module": "mental_health", "task": "multiclass", "version": version, "model_name": best,
        "vectorizer": vec, "model": fitted[best], "explainer": explainer, "labels": labels,
        "risk_mapping": RISK_LEVEL, "test_metrics": results[best]["test"], "known_words": known_words,
    }
    save_bundle("mental_health", bundle, report, force=args.force, log=log)
    log(f"[mental_health] done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
