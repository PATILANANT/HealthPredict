"""Saving / loading of model bundles with 'promote only if better' logic (FR-12, FR-16)."""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import joblib

from ml.config import MODELS_DIR


def bundle_path(module: str, models_dir: Path = MODELS_DIR) -> Path:
    return Path(models_dir) / f"{module}.joblib"


def metrics_path(module: str, models_dir: Path = MODELS_DIR) -> Path:
    return Path(models_dir) / f"{module}_metrics.json"


def load_metrics(module: str, models_dir: Path = MODELS_DIR) -> dict | None:
    p = metrics_path(module, models_dir)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def save_bundle(module: str, bundle: dict, report: dict, force: bool = False,
                models_dir: Path = MODELS_DIR, log=print) -> bool:
    """Save the bundle as the live model if it is at least as good as the current one.

    Returns True when the new model was promoted. A model that is not promoted is kept as
    `<module>.candidate.joblib` so it can be inspected.
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    current = load_metrics(module, models_dir)
    new_score = report["selection_score"]
    promoted = True
    if current is not None and not force:
        old_score = current.get("selection_score", -1)
        if new_score + 1e-9 < old_score:
            promoted = False
            log(f"New model scores {new_score:.4f} < current {old_score:.4f} "
                f"on {report['selection_metric']} -> NOT PROMOTED")

    report = dict(report, promoted=promoted)
    if promoted:
        live = bundle_path(module, models_dir)
        if live.exists():
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            archive = models_dir / "archive"
            archive.mkdir(exist_ok=True)
            shutil.copy2(live, archive / f"{module}-{stamp}.joblib")
        joblib.dump(bundle, live, compress=3)
        metrics_path(module, models_dir).write_text(json.dumps(report, indent=2), encoding="utf-8")
        log(f"Saved {live.name} ({report['best_model']}) -> PROMOTED")
    else:
        joblib.dump(bundle, models_dir / f"{module}.candidate.joblib", compress=3)
        (models_dir / f"{module}.candidate_metrics.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")
    return promoted
