from src.data_loader import load_data
from src.preprocessing import clean_data
from src.feature_engineering import engineer_features
from src.trainer import (
    prepare_training_data,
    train_all_models,
)
import joblib
import shutil
from src.config import MODELS_DIR

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
print(feature_columns)
results, best_model = train_all_models(
    X_train,
    X_test,
    y_train,
    y_test,
)

joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
joblib.dump(feature_columns, MODELS_DIR / "feature_column.pkl")

# Keep the production artifact synchronized with the freshly trained model.
shutil.copy2(
    MODELS_DIR / "best_model.pkl",
    MODELS_DIR / "production_model.pkl",
)

print("\nModel Comparison\n")

print(results)
results.to_csv(
    "reports/model_comparison/model_results.csv",
    index=False,
)