"""Prediction engine using the exact scaler fitted during model training."""

from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "trained_models"
MODEL_PATH = MODEL_DIR / "production_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
FEATURES_PATH = MODEL_DIR / "feature_column.pkl"

for required_path in (MODEL_PATH, SCALER_PATH, FEATURES_PATH):
    if not required_path.exists():
        raise FileNotFoundError(
            f"Required model artifact not found: {required_path}. "
            "Run the training and model-promotion scripts first."
        )

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
FEATURE_COLUMNS = joblib.load(FEATURES_PATH)


def predict_house_price(features: dict) -> float:
    """Predict a house price from raw property inputs.

    The input dictionary is not modified. The saved training scaler is applied
    before prediction, preserving the training/inference preprocessing contract.
    """
    required = set(FEATURE_COLUMNS)
    derived = {"house_age", "is_renovated", "renovation_age", "bath_bed_ratio"}
    raw_required = required - derived
    missing = sorted(raw_required - set(features))
    if missing:
        raise ValueError(f"Missing required input features: {', '.join(missing)}")

    input_data = dict(features)
    bedrooms = float(input_data["bedrooms"])
    bathrooms = float(input_data["bathrooms"])
    built_year = int(input_data["yr_built"])
    renovated_year = int(input_data["yr_renovated"])
    sale_year = int(input_data["sale_year"])

    if bedrooms < 0 or bathrooms < 0:
        raise ValueError("Bedrooms and bathrooms cannot be negative.")
    if built_year > sale_year:
        raise ValueError("Year built cannot be later than the sale year.")
    if renovated_year and renovated_year > sale_year:
        raise ValueError("Year renovated cannot be later than the sale year.")

    input_data.update({
        "house_age": max(sale_year - built_year, 0),
        "is_renovated": int(renovated_year > 0),
        "renovation_age": max(sale_year - renovated_year, 0) if renovated_year else 0,
        "bath_bed_ratio": bathrooms / bedrooms if bedrooms else 0.0,
    })

    input_df = pd.DataFrame([input_data]).reindex(columns=FEATURE_COLUMNS)
    input_df = input_df.apply(pd.to_numeric, errors="raise")

    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]
    return float(prediction)
