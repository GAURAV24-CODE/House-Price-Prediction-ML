import matplotlib.pyplot as plt
import pandas as pd
import joblib

from src.config import MODELS_DIR, FIGURES_DIR, CSV_DIR


def generate_feature_importance(feature_names):
    """
    Generate feature importance report from the best model.
    """

    model = joblib.load(
        MODELS_DIR / "best_model.pkl"
    )

    # Tree-based models expose feature_importances_
    importance = model.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    # Save CSV
    importance_df.to_csv(
        CSV_DIR / "feature_importance.csv",
        index=False
    )

    # Plot
    plt.figure(figsize=(10, 8))

    bars = plt.barh(
        importance_df["Feature"],
        importance_df["Importance"]
    )

    plt.gca().invert_yaxis()

    # Highlight top feature
    bars[0].set_color("green")

    plt.xlabel("Importance Score")
    plt.title("Feature Importance (XGBoost)")

    plt.grid(axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "feature_importance.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return importance_df