"""Generate PNG figures for the project report / PPT from the saved metrics.

Usage:  python -m ml.figures      -> writes reports/figures/*.png and reports/results.md
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ml.common.store import load_metrics  # noqa: E402
from ml.config import ROOT  # noqa: E402
from ml.schemas import MODULES  # noqa: E402

OUT = ROOT / "reports" / "figures"


def comparison_chart(module: str, rep: dict):
    names = sorted(rep["models"], key=lambda n: rep["models"][n]["test"]["f1"])
    metrics = ["accuracy", "precision", "recall", "f1"] + (["roc_auc"] if rep["task"] == "binary" else [])
    y = np.arange(len(names))
    h = 0.8 / len(metrics)
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(names) + 1.5))
    for i, m in enumerate(metrics):
        ax.barh(y + i * h, [rep["models"][n]["test"][m] * 100 for n in names], h, label=m.upper().replace("_", "-"))
    ax.set_yticks(y + h * (len(metrics) - 1) / 2, [n + (" *" if n == rep["best_model"] else "") for n in names])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Test-set score (%)")
    ax.set_title(f"{MODULES[module]['title']}: classifier comparison (* = selected model)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / f"{module}_comparison.png", dpi=150)
    plt.close(fig)


def confusion_chart(module: str, rep: dict):
    t = rep["models"][rep["best_model"]]["test"]
    cm = np.array(t["confusion"])
    labels = t.get("labels") or ([MODULES[module]["negative_label"], MODULES[module]["positive_label"]])
    fig, ax = plt.subplots(figsize=(1.1 * len(labels) + 2.5, 1.0 * len(labels) + 2))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{MODULES[module]['title']}: {rep['best_model']}")
    fig.tight_layout()
    fig.savefig(OUT / f"{module}_confusion.png", dpi=150)
    plt.close(fig)


def results_markdown(reports: dict) -> str:
    lines = ["# Results (held-out test sets)\n"]
    for module, rep in reports.items():
        binary = rep["task"] == "binary"
        lines.append(f"## {MODULES[module]['title']}\n")
        lines.append(f"Best model: **{rep['best_model']}** (selected on {rep['selection_metric']}), "
                     f"train {rep['n_train']} / test {rep['n_test']}.\n")
        head = "| Classifier | Accuracy | Precision | Recall | F1 |" + (" Specificity | ROC-AUC |" if binary else " Weighted F1 |")
        lines += [head, "|" + "---|" * (head.count("|") - 1)]
        for n, r in sorted(rep["models"].items(), key=lambda kv: -kv[1]["test"]["f1"]):
            t = r["test"]
            row = f"| {n} | {t['accuracy']*100:.2f} | {t['precision']*100:.2f} | {t['recall']*100:.2f} | {t['f1']*100:.2f} |"
            row += (f" {t['specificity']*100:.2f} | {t['roc_auc']*100:.2f} |" if binary else f" {t['f1_weighted']*100:.2f} |")
            lines.append(row)
        if rep.get("distress_binary_test"):
            d = rep["distress_binary_test"]
            lines.append(f"\nDistress vs Normal: accuracy {d['accuracy']*100:.2f}%, precision {d['precision']*100:.2f}%, "
                         f"recall {d['recall']*100:.2f}%, F1 {d['f1']*100:.2f}%.")
        lines.append("")
    return "\n".join(lines)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    reports = {m: r for m in MODULES if (r := load_metrics(m))}
    for module, rep in reports.items():
        comparison_chart(module, rep)
        confusion_chart(module, rep)
    (ROOT / "reports" / "results.md").write_text(results_markdown(reports), encoding="utf-8")
    print(f"Wrote figures for {list(reports)} to {OUT}")


if __name__ == "__main__":
    main()
