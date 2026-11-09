import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import cross_validate

from src.data_loader import load_data
from src.preprocessing import clean_data
from src.feature_engineering import engineer_features

from src.config import (
    MODELS_DIR,
    CSV_DIR,
    FIGURES_DIR,
)


# -------------------------------
# Load Dataset
# -------------------------------

df = load_data()

df = clean_data(df)

df = engineer_features(df)


# -------------------------------
# Prepare Features
# -------------------------------

X = df.drop(columns=["price"])

y = df["price"]

if "date" in X.columns:

    X["sale_year"] = X["date"].dt.year

    X["sale_month"] = X["date"].dt.month

    X.drop(columns=["date"], inplace=True)


# -------------------------------
# Load Best Model
# -------------------------------

model = joblib.load(
    MODELS_DIR / "best_model.pkl"
)


# -------------------------------
# Cross Validation
# -------------------------------

scoring = {

    "R2": "r2",

    "RMSE": "neg_root_mean_squared_error"

}

scores = cross_validate(

    model,

    X,

    y,

    cv=5,

    scoring=scoring,

    return_train_score=False

)


# -------------------------------
# Create Results DataFrame
# -------------------------------

results = pd.DataFrame({

    "Fold": range(1, 6),

    "R2": scores["test_R2"],

    "RMSE": -scores["test_RMSE"]

})


# -------------------------------
# Statistics
# -------------------------------

summary = pd.DataFrame({

    "Metric": [

        "Average R2",

        "Std R2",

        "Average RMSE",

        "Std RMSE"

    ],

    "Value": [

        results["R2"].mean(),

        results["R2"].std(),

        results["RMSE"].mean(),

        results["RMSE"].std()

    ]

})


# -------------------------------
# Save CSV
# -------------------------------

results.to_csv(

    CSV_DIR / "cross_validation_results.csv",

    index=False

)

summary.to_csv(

    CSV_DIR / "cross_validation_summary.csv",

    index=False

)


# -------------------------------
# Plot R²
# -------------------------------

plt.figure(figsize=(8,5))

plt.plot(

    results["Fold"],

    results["R2"],

    marker="o",

    linewidth=2

)

plt.title("5-Fold Cross Validation (R²)")

plt.xlabel("Fold")

plt.ylabel("R²")

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(

    FIGURES_DIR /

    "cross_validation_r2.png",

    dpi=300,

    bbox_inches="tight"

)

plt.close()


# -------------------------------
# Plot RMSE
# -------------------------------

plt.figure(figsize=(8,5))

plt.plot(

    results["Fold"],

    results["RMSE"],

    marker="o",

    linewidth=2

)

plt.title("5-Fold Cross Validation (RMSE)")

plt.xlabel("Fold")

plt.ylabel("RMSE")

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(

    FIGURES_DIR /

    "cross_validation_rmse.png",

    dpi=300,

    bbox_inches="tight"

)

plt.close()


# -------------------------------
# Print Results
# -------------------------------

print("\nCross Validation Results\n")

print(results)

print("\nSummary\n")

print(summary)