import pandas as pd
import joblib

from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV

from src.data_loader import load_data
from src.preprocessing import clean_data
from src.feature_engineering import engineer_features
from src.trainer import prepare_training_data

from src.config import MODELS_DIR


# ------------------------
# Load Dataset
# ------------------------

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


# ------------------------
# Parameter Space
# ------------------------

param_grid = {

    "n_estimators": [100, 200, 300],

    "max_depth": [3, 5, 7, 9],

    "learning_rate": [0.01, 0.05, 0.1],

    "subsample": [0.8, 0.9, 1.0],

    "colsample_bytree": [0.8, 0.9, 1.0]

}


# ------------------------
# Model
# ------------------------

model = XGBRegressor(
    random_state=42
)


# ------------------------
# Random Search
# ------------------------

search = RandomizedSearchCV(

    estimator=model,

    param_distributions=param_grid,

    n_iter=20,

    scoring="r2",

    cv=5,

    verbose=2,

    random_state=42,

    n_jobs=-1

)

search.fit(
    X_train,
    y_train
)


# ------------------------
# Best Model
# ------------------------

best_model = search.best_estimator_

print("\nBest Parameters\n")

print(search.best_params_)

print("\nBest Cross Validation Score\n")

print(search.best_score_)


# ------------------------
# Save Tuned Model
# ------------------------

joblib.dump(
    best_model,
    MODELS_DIR / "best_model_tuned.pkl"
)

print("\nTuned model saved successfully.")