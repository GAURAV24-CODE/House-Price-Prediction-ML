import pandas as pd

from src.feature_selection import select_features

importance_df = pd.read_csv(
    "reports/csv/feature_importance.csv"
)

selected_features = select_features(
    importance_df,
    threshold=0.01
)

print("\nSelected Features:")

for feature in selected_features:
    print(feature)