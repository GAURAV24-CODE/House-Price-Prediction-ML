from src.data_loader import load_data
from src.preprocessing import clean_data
from src.feature_engineering import engineer_features

df = load_data()

df = clean_data(df)

df = engineer_features(df)

print(df.head())

print()

print(df.columns.tolist())