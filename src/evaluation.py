import time
import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def evaluate_model(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
):
    """
    Train and evaluate a regression model.
    """

    # ---------------------------
    # Training Time
    # ---------------------------
    start_train = time.time()

    model.fit(X_train, y_train)

    end_train = time.time()

    train_time = end_train - start_train

    # ---------------------------
    # Prediction Time
    # ---------------------------
    start_pred = time.time()

    predictions = model.predict(X_test)

    end_pred = time.time()

    prediction_time = end_pred - start_pred

    # ---------------------------
    # Metrics
    # ---------------------------
    mae = mean_absolute_error(y_test, predictions)

    mse = mean_squared_error(y_test, predictions)

    rmse = np.sqrt(mse)

    r2 = r2_score(y_test, predictions)

    n = len(y_test)

    p = X_test.shape[1]

    adjusted_r2 = 1 - (
        (1 - r2) * (n - 1)
    ) / (n - p - 1)

    return {

        "MAE": mae,

        "MSE": mse,

        "RMSE": rmse,

        "R2": r2,

        "Adjusted R2": adjusted_r2,

        "Training Time": train_time,

        "Prediction Time": prediction_time,
    }