from src.data_loader import load_data
from src.preprocessing import clean_data
from src.feature_engineering import engineer_features
from src.trainer import prepare_training_data

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

print("Training Shape:", X_train.shape)

print("Testing Shape:", X_test.shape)

print("Target Shape:", y_train.shape)