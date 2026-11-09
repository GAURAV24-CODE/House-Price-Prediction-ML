import shutil
import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)

from src.data_loader import load_data
from src.preprocessing import clean_data
from src.feature_engineering import engineer_features
from src.trainer import prepare_training_data

from src.config import (
    MODELS_DIR,
    CSV_DIR,
)

# ------------------------------------
# Load Dataset
# ------------------------------------

df = load_data()
df = clean_data(df)
df = engineer_features(df)

(
    X_train,
    X_test,
    y_train,
    y_test,
    scaler,
    feature_columns,
) = prepare_training_data(df)

# ------------------------------------
# Load Models
# ------------------------------------

baseline_model = joblib.load(
    MODELS_DIR / "best_model.pkl"
)

tuned_model = joblib.load(
    MODELS_DIR / "best_model_tuned.pkl"
)

# ------------------------------------
# Predictions
# ------------------------------------

baseline_pred = baseline_model.predict(X_test)

tuned_pred = tuned_model.predict(X_test)

# ------------------------------------
# Evaluation Function
# ------------------------------------

def evaluate_model(name, y_true, y_pred):

    return {
        "Model": name,
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(
            mean_squared_error(
                y_true,
                y_pred
            )
        ),
    }

results = [
    evaluate_model(
        "Baseline",
        y_test,
        baseline_pred,
    ),
    evaluate_model(
        "Tuned",
        y_test,
        tuned_pred,
    ),
]

results_df = pd.DataFrame(results)

print("\nModel Comparison\n")
print(results_df)

# ------------------------------------
# Save Benchmark Report
# ------------------------------------

results_df.to_csv(
    CSV_DIR / "benchmark_results.csv",
    index=False
)

# ------------------------------------
# Promote Better Model
# ------------------------------------

baseline_r2 = results_df.loc[
    results_df["Model"] == "Baseline",
    "R2"
].values[0]

tuned_r2 = results_df.loc[
    results_df["Model"] == "Tuned",
    "R2"
].values[0]

if tuned_r2 > baseline_r2:

    shutil.copy(
        MODELS_DIR / "best_model_tuned.pkl",
        MODELS_DIR / "production_model.pkl",
    )

    winner = "Tuned Model"

else:

    shutil.copy(
        MODELS_DIR / "best_model.pkl",
        MODELS_DIR / "production_model.pkl",
    )

    winner = "Baseline Model"

print(f"\nProduction Model : {winner}")
print("\nproduction_model.pkl updated successfully.")