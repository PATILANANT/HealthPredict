"""Picklable preprocessing functions used inside saved model pipelines.

They live in their own module so that a joblib bundle can always be re-loaded by the
web app (pickle stores the import path of each function).
"""
import numpy as np
import pandas as pd

DIABETES_ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def diabetes_zeros_to_nan(X: pd.DataFrame) -> pd.DataFrame:
    """In the Pima file a 0 in these columns means 'not measured', not a real value."""
    X = pd.DataFrame(X).copy()
    for col in DIABETES_ZERO_AS_MISSING:
        if col in X.columns:
            X[col] = X[col].replace(0, np.nan)
    return X
