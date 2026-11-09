from src.data_loader import load_data
from src.preprocessing import clean_data
from src.feature_engineering import engineer_features
from src.trainer import prepare_training_data

from src.model import get_models
from src.evaluation import evaluate_model


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

models = get_models()

linear = models["Linear Regression"]

results = evaluate_model(
    linear,
    X_train,
    X_test,
    y_train,
    y_test,
)

print(results)