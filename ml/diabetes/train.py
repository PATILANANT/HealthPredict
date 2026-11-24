"""Train and benchmark the Diabetes module on the Pima Indians Diabetes Database.

Usage:  python -m ml.diabetes.train [--force] [--data path.csv]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from ml.config import DATA_FILES, REQUIRED_COLUMNS
from ml.common.tabular import train_tabular
from ml.common.transformers import DIABETES_ZERO_AS_MISSING, diabetes_zeros_to_nan
from ml.schemas import model_features

TARGET = "Outcome"


def load(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(REQUIRED_COLUMNS["diabetes"]) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    df = df[REQUIRED_COLUMNS["diabetes"]].drop_duplicates()
    return df.dropna(subset=[TARGET])


def age_band(age) -> str:
    return "21-30" if age <= 30 else ("31-45" if age <= 45 else "46+")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA_FILES["diabetes"]))
    ap.add_argument("--force", action="store_true", help="promote even if not better")
    args = ap.parse_args(argv)

    df = load(args.data)
    features = model_features("diabetes")
    X, y = df[features], df[TARGET].astype(int)

    preprocessor = Pipeline([
        ("zeros_to_missing", FunctionTransformer(diabetes_zeros_to_nan)),
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    cleaned = diabetes_zeros_to_nan(X)
    reference = {c: float(cleaned[c].median()) for c in features}
    info = {
        "file": str(args.data), "rows": int(len(df)), "positive_rate": round(float(y.mean()), 4),
        "missing_after_zero_fix": {c: int(cleaned[c].isna().sum()) for c in DIABETES_ZERO_AS_MISSING},
    }
    train_tabular(
        module="diabetes", X=X, y=y, preprocessor=preprocessor, use_smote=False, beta=1.0,
        select_metric="f1", reference=reference, dataset_info=info, force=args.force,
        groups_fn=lambda Xt: {"age_band": np.array([age_band(a) for a in Xt["Age"]])},
    )


if __name__ == "__main__":
    main()
