"""Regression tests for shared feature engineering and prediction preprocessing."""

import numpy as np
import pandas as pd

from src.feature_engineering import engineer_features
from src.predictor import predict_house_price, scaler, FEATURE_COLUMNS


def test_feature_engineering_uses_sale_year():
    sample = pd.DataFrame({
        "date": ["2015-05-01"],
        "yr_built": [2000],
        "yr_renovated": [2010],
        "bedrooms": [2],
        "bathrooms": [1.5],
    })

    result = engineer_features(sample)

    assert result.loc[0, "sale_year"] == 2015
    assert result.loc[0, "sale_month"] == 5
    assert result.loc[0, "house_age"] == 15
    assert result.loc[0, "renovation_age"] == 5
    assert result.loc[0, "bath_bed_ratio"] == 0.75
    assert "date" not in result.columns


def test_saved_scaler_matches_model_feature_count():
    assert scaler.n_features_in_ == len(FEATURE_COLUMNS)


def test_prediction_accepts_raw_features_without_mutating_input():
    features = {
        "bedrooms": 3, "bathrooms": 2.0, "sqft_living": 2000,
        "sqft_lot": 5000, "floors": 1.0, "waterfront": 0, "view": 0,
        "condition": 3, "grade": 7, "sqft_above": 1500,
        "sqft_basement": 500, "yr_built": 1995, "yr_renovated": 0,
        "zipcode": 98052, "lat": 47.60, "long": -122.20,
        "sqft_living15": 2000, "sqft_lot15": 5000,
        "sale_year": 2015, "sale_month": 5,
    }

    prediction = predict_house_price(features)

    assert np.isfinite(prediction)
    assert prediction > 0
    assert "house_age" not in features


def test_prediction_rejects_missing_features():
    try:
        predict_house_price({"bedrooms": 3})
    except ValueError as exc:
        assert "Missing required input features" in str(exc)
    else:
        raise AssertionError("Expected missing features to raise ValueError")
