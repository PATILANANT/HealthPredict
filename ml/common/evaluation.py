"""Evaluation helpers: accuracy, precision, recall, F1, specificity, ROC-AUC (Chapter 5.5)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             fbeta_score, precision_score, recall_score, roc_auc_score)


def _r(x: float) -> float:
    return round(float(x), 4)


def binary_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    y_pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": _r(accuracy_score(y_true, y_pred)),
        "precision": _r(precision_score(y_true, y_pred, zero_division=0)),
        "recall": _r(recall_score(y_true, y_pred, zero_division=0)),
        "f1": _r(f1_score(y_true, y_pred, zero_division=0)),
        "f2": _r(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        "specificity": _r(tn / (tn + fp) if (tn + fp) else 0.0),
        "roc_auc": _r(roc_auc_score(y_true, proba)) if len(set(y_true)) > 1 else None,
        "pr_auc": _r(average_precision_score(y_true, proba)) if len(set(y_true)) > 1 else None,
        "threshold": _r(threshold),
        "confusion": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def best_threshold(y_true, proba, beta: float = 1.0) -> float:
    """Pick the decision threshold that maximises F-beta on out-of-fold predictions.

    beta=1 balances precision and recall; beta=2 favours recall (missing a stroke is worse
    than a false alarm for a screening tool).
    """
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    scores = [fbeta_score(y_true, (proba >= t).astype(int), beta=beta, zero_division=0)
              for t in grid]
    return float(grid[int(np.argmax(scores))])


def multiclass_metrics(y_true, y_pred, labels: list[str]) -> dict:
    per_class = {}
    p = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    r = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    for i, lab in enumerate(labels):
        per_class[lab] = {"precision": _r(p[i]), "recall": _r(r[i]), "f1": _r(f[i]),
                          "support": int(np.sum(np.asarray(y_true) == lab))}
    return {
        "accuracy": _r(accuracy_score(y_true, y_pred)),
        "precision": _r(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": _r(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1": _r(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": _r(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class": per_class,
        "labels": labels,
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def group_metrics(y_true, proba, threshold: float, groups) -> dict:
    """Fairness check: recall / precision / positive-rate per subgroup on the test set."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(proba) >= threshold).astype(int)
    groups = np.asarray(groups)
    out = {}
    for g in sorted(set(groups.tolist()), key=str):
        m = groups == g
        out[str(g)] = {
            "n": int(m.sum()),
            "positives": int(y_true[m].sum()),
            "recall": _r(recall_score(y_true[m], y_pred[m], zero_division=0)),
            "precision": _r(precision_score(y_true[m], y_pred[m], zero_division=0)),
            "flagged_rate": _r(y_pred[m].mean() if m.sum() else 0.0),
        }
    return out
