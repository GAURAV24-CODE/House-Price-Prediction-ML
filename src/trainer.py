import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def prepare_training_data(df: pd.DataFrame):
    """
    Prepare dataset for model training.
    """

    # -----------------------
    # Target
    # -----------------------
    y = df["price"]

    # -----------------------
    # Features
    # -----------------------
    X = df.drop(columns=["price"])

    # Date-derived fields are created by engineer_features so training and
    # inference share one implementation.
    if "date" in X.columns:
        raise ValueError("Unexpected 'date' column. Run engineer_features(df) first.")

    # -----------------------
    # Train Test Split
    # -----------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    # -----------------------
    # Scaling
    # -----------------------
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    return (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        scaler,
        X.columns.tolist(),
    )

import pandas as pd
import joblib

from src.model import get_models
from src.evaluation import evaluate_model
from src.config import MODELS_DIR


def train_all_models(
    X_train,
    X_test,
    y_train,
    y_test,
):
    """
    Train every regression model and compare results.
    """

    models = get_models()

    results = []

    best_model = None

    best_score = float("-inf")

    for name, model in models.items():

        print(f"Training {name}...")

        metrics = evaluate_model(
            model,
            X_train,
            X_test,
            y_train,
            y_test,
        )

        metrics["Model"] = name

        results.append(metrics)

        # Select best model using R2
        if metrics["R2"] > best_score:

            best_score = metrics["R2"]

            best_model = model

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="R2",
        ascending=False,
    )


    joblib.dump(
        best_model,
        MODELS_DIR / "best_model.pkl"
    )

    return results_df, best_model