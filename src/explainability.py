import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt

from src.config import (
    MODELS_DIR,
    FIGURES_DIR,
    CSV_DIR,
)


def generate_shap_report(
    X_train,
    X_test,
    feature_columns,
):
    """
    Generate SHAP explainability report.
    """

    print("\nLoading production model...")

    model = joblib.load(
        MODELS_DIR / "production_model.pkl"
    )

    print("Creating SHAP Explainer...")

    explainer = shap.Explainer(model)

    # Use first 200 samples for faster SHAP computation
    X_sample = X_test.iloc[:200].copy()

    print("Calculating SHAP values...")

    shap_values = explainer(X_sample)

    # ==================================================
    # Feature Importance CSV
    # ==================================================

    importance = pd.DataFrame({
        "Feature": X_sample.columns,
        "Mean_SHAP_Value": abs(shap_values.values).mean(axis=0)
    })

    importance = (
        importance
        .sort_values(
            by="Mean_SHAP_Value",
            ascending=False
        )
        .reset_index(drop=True)
    )

    importance.to_csv(
        CSV_DIR / "shap_feature_importance.csv",
        index=False
    )

    top_feature = importance.iloc[0]["Feature"]

    # ==================================================
    # SHAP Summary Plot
    # ==================================================

    print("Generating SHAP Summary Plot...")

    plt.figure(figsize=(16, 9))

    shap.plots.beeswarm(
        shap_values,
        max_display=len(feature_columns),
        show=False
    )

    plt.title(
        "SHAP Summary Plot",
        fontsize=18,
        pad=20
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "shap_summary.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ==================================================
    # SHAP Feature Importance
    # ==================================================

    print("Generating SHAP Feature Importance Plot...")

    plt.figure(figsize=(16, 9))

    shap.plots.bar(
        shap_values,
        max_display=len(feature_columns),
        show=False
    )

    plt.title(
        "Global Feature Importance",
        fontsize=18,
        pad=20
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "shap_bar.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ==================================================
    # SHAP Waterfall Plot
    # ==================================================

    print("Generating SHAP Waterfall Plot...")

    plt.figure(figsize=(16, 9))

    shap.plots.waterfall(
        shap_values[0],
        max_display=len(feature_columns),
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "shap_waterfall.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ==================================================
    # SHAP Dependence Plot
    # ==================================================

    print(f"Generating Dependence Plot for: {top_feature}")

    plt.figure(figsize=(14, 8))

    shap.dependence_plot(
        top_feature,
        shap_values.values,
        X_sample,
        interaction_index="auto",
        show=False
    )

    plt.title(
        f"SHAP Dependence Plot - {top_feature}",
        fontsize=18,
        pad=20
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR /
        f"shap_dependence_{top_feature}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\n" + "=" * 60)
    print("SHAP REPORT GENERATED SUCCESSFULLY")
    print("=" * 60)

    print(f"\nTop Feature : {top_feature}")

    print("\nGenerated Files:")

    print("✓ shap_summary.png")
    print("✓ shap_bar.png")
    print("✓ shap_waterfall.png")
    print(f"✓ shap_dependence_{top_feature}.png")
    print("✓ shap_feature_importance.csv")

    return shap_values