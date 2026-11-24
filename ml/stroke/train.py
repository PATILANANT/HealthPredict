"""Train and benchmark the Brain Stroke module on the Kaggle Stroke Prediction Dataset.

Only ~5% of records are stroke cases, so SMOTE is applied inside the pipeline (training folds
only) and the threshold is tuned for F2 (recall-weighted) - a screening tool should rather
raise a false alarm than miss a stroke.

Usage:  python -m ml.stroke.train [--force] [--data path.csv]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.config import DATA_FILES, REQUIRED_COLUMNS
from ml.common.tabular import default_reference, train_tabular
from ml.schemas import model_features

TARGET = "stroke"
NUMERIC = ["age", "avg_glucose_level", "bmi"]
BINARY = ["hypertension", "heart_disease"]
CATEGORICAL = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]


def load(path) -> pd.DataFrame:
    df = pd.read_csv(path, na_values=["N/A", "NA", ""])
    missing = set(REQUIRED_COLUMNS["stroke"]) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    df = df[REQUIRED_COLUMNS["stroke"]].drop_duplicates()
    return df.dropna(subset=[TARGET])


def age_band(age) -> str:
    return "<40" if age < 40 else ("40-59" if age < 60 else "60+")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA_FILES["stroke"]))
    ap.add_argument("--force", action="store_true", help="promote even if not better")
    args = ap.parse_args(argv)

    df = load(args.data)
    features = model_features("stroke")
    X, y = df[features].copy(), df[TARGET].astype(int)
    for c in BINARY:
        X[c] = X[c].astype(int)

    preprocessor = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), NUMERIC),
        ("bin", "passthrough", BINARY),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])
    info = {"file": str(args.data), "rows": int(len(df)), "positive_rate": round(float(y.mean()), 4),
            "bmi_missing": int(X["bmi"].isna().sum())}
    train_tabular(
        module="stroke", X=X, y=y, preprocessor=preprocessor, use_smote=True, beta=2.0,
        select_metric="f2", reference=default_reference(X), dataset_info=info, force=args.force,
        groups_fn=lambda Xt: {"gender": Xt["gender"].to_numpy(),
                              "age_band": np.array([age_band(a) for a in Xt["age"]])},
    )


if __name__ == "__main__":
    main()
