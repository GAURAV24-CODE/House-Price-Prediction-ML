"""
Prediction Script

This script loads the production model and predicts
the house price for a sample input.
"""

from src.predictor import predict_house_price


def main():
    # --------------------------------------------------
    # Example House Features
    # --------------------------------------------------

    house_features = {
        "bedrooms": 4,
        "bathrooms": 3.0,
        "sqft_living": 2500,
        "sqft_lot": 5000,
        "floors": 2,
        "waterfront": 0,
        "view": 2,
        "condition": 3,
        "grade": 8,
        "sqft_above": 2000,
        "sqft_basement": 500,
        "yr_built": 1995,
        "yr_renovated": 0,
        "zipcode": 98052,
        "lat": 47.641,
        "long": -122.128,
        "sqft_living15": 2400,
        "sqft_lot15": 5200,
        "house_age": 30,
        "renovation_age": 0,
        "sale_year": 2015,
        "sale_month": 5,
    }

    predicted_price = predict_house_price(house_features)

    print("=" * 50)
    print("HOUSE PRICE PREDICTION")
    print("=" * 50)
    print(f"\nPredicted Price : ${predicted_price:,.2f}")


if __name__ == "__main__":
    main()