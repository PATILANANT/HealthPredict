"""Generic, leakage-free training loop for the clinical modules (Diabetes, Stroke).

* Stratified 80/20 train/test split; the test set is used ONLY for the final report.
* Every preprocessing step (imputation, scaling, encoding, SMOTE) lives inside the pipeline,
  so it is re-fitted on each cross-validation fold and never sees validation/test rows.
* The decision threshold and the best model are chosen on 5-fold out-of-fold predictions
  of the training set, not on the test set.
"""
from __future__ import annotations

import time
from datetime import datetime

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline

from ml.config import RANDOM_STATE, TEST_SIZE
from ml.common.evaluation import best_threshold, binary_metrics, group_metrics
from ml.common.store import save_bundle
from ml.common.zoo import tabular_models


def build_pipeline(preprocessor, model, use_smote: bool):
    prep = clone(preprocessor)
    # imblearn does not allow a nested sklearn Pipeline as a step, so flatten it.
    steps = list(prep.steps) if isinstance(prep, Pipeline) else [("prep", prep)]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=RANDOM_STATE)))
    steps.append(("clf", clone(model)))
    return ImbPipeline(steps)


def default_reference(X: pd.DataFrame) -> dict:
    ref = {}
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]) and X[c].nunique() > 2:
            ref[c] = float(X[c].median())
        else:
            ref[c] = X[c].mode().iloc[0]
            if isinstance(ref[c], (np.integer,)):
                ref[c] = int(ref[c])
    return ref


def train_tabular(*, module: str, X: pd.DataFrame, y: pd.Series, preprocessor, use_smote: bool,
                  beta: float, select_metric: str, groups_fn=None, reference: dict | None = None,
                  dataset_info: dict | None = None, force: bool = False,
                  only: list[str] | None = None, log=print) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    log(f"[{module}] train={len(X_train)} test={len(X_test)} positives(train)={int(y_train.sum())}")

    results, fitted = {}, {}
    for name, model in tabular_models(RANDOM_STATE).items():
        if only and name not in only:
            continue
        t0 = time.time()
        pipe = build_pipeline(preprocessor, model, use_smote)
        oof = cross_val_predict(pipe, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
        thr = best_threshold(y_train, oof, beta=beta)
        pipe.fit(X_train, y_train)
        p_test = pipe.predict_proba(X_test)[:, 1]
        results[name] = {
            "cv": binary_metrics(y_train, oof, thr),
            "test": binary_metrics(y_test, p_test, thr),
            "test_default_threshold": binary_metrics(y_test, p_test, 0.5),
            "seconds": round(time.time() - t0, 1),
        }
        fitted[name] = (pipe, thr, p_test)
        t = results[name]["test"]
        log(f"  {name:<22} cv-{select_metric}={results[name]['cv'][select_metric]:.3f} "
            f"| test acc={t['accuracy']:.3f} prec={t['precision']:.3f} rec={t['recall']:.3f} "
            f"f1={t['f1']:.3f} auc={t['roc_auc']:.3f} thr={thr:.2f} ({results[name]['seconds']}s)")

    best = max(results, key=lambda n: (results[n]["cv"][select_metric], results[n]["cv"]["roc_auc"]))
    pipe, thr, p_test = fitted[best]
    log(f"[{module}] best model: {best} (cv {select_metric}={results[best]['cv'][select_metric]:.4f})")

    fairness = {}
    if groups_fn is not None:
        for gname, groups in groups_fn(X_test).items():
            fairness[gname] = group_metrics(y_test, p_test, thr, groups)

    version = datetime.now().strftime("%Y.%m.%d-%H%M%S")
    report = {
        "module": module,
        "task": "binary",
        "version": version,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "best_model": best,
        "selection_metric": f"cv_{select_metric}",
        "selection_score": results[best]["cv"][select_metric],
        "threshold": thr,
        "threshold_beta": beta,
        "smote": use_smote,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "dataset": dataset_info or {},
        "models": results,
        "fairness": fairness,
    }
    bundle = {
        "module": module,
        "task": "binary",
        "version": version,
        "model_name": best,
        "pipeline": pipe,
        "threshold": thr,
        "features": list(X.columns),
        "reference": reference or default_reference(X_train),
        "test_metrics": results[best]["test"],
    }
    report["promoted"] = save_bundle(module, bundle, report, force=force, log=log)
    return report
