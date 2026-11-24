"""Project-wide paths and constants shared by training scripts and the web app."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
MODELS_DIR = ROOT / "models"
RANDOM_STATE = 42
TEST_SIZE = 0.20

DATA_FILES = {
    "diabetes": DATA_DIR / "diabetes.csv",
    "stroke": DATA_DIR / "stroke.csv",
    "mental_health": DATA_DIR / "mental_health.csv",
}

# Columns a CSV must contain before it can be used to (re)train a module.
REQUIRED_COLUMNS = {
    "diabetes": ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
                 "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"],
    "stroke": ["gender", "age", "hypertension", "heart_disease", "ever_married", "work_type",
               "Residence_type", "avg_glucose_level", "bmi", "smoking_status", "stroke"],
    "mental_health": ["statement", "status"],
}
