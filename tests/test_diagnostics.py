from joblib import load

from src.data_loader import load_data
from src.preprocessing import clean_data
from src.feature_engineering import engineer_features
from src.trainer import prepare_training_data

from src.config import MODELS_DIR

from src.diagnostics import (
    plot_actual_vs_predicted,
    plot_residuals,
    plot_residual_distribution,
    plot_prediction_error_distribution,
)

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

model = load(MODELS_DIR / "best_model.pkl")

predictions = model.predict(X_test)

plot_actual_vs_predicted(y_test, predictions)
plot_residuals(y_test, predictions)
plot_residual_distribution(y_test, predictions)
plot_prediction_error_distribution(y_test, predictions)

print("Diagnostics generated successfully!")