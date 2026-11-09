from joblib import load

from src.feature_importance import generate_feature_importance

from src.config import MODELS_DIR

feature_names = load(
    MODELS_DIR / "feature_column.pkl"
)

importance_df = generate_feature_importance(
    feature_names
)

print(importance_df.head(10))