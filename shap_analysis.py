"""
SHAP Analysis Runner

This script:
1. Loads the dataset
2. Cleans the data
3. Performs feature engineering
4. Prepares training and testing datasets
5. Generates SHAP explainability reports
"""

from src.data_loader import load_data
from src.preprocessing import clean_data
from src.feature_engineering import engineer_features
from src.trainer import prepare_training_data
from src.explainability import generate_shap_report
import pandas as pd

def main():
    print("=" * 60)
    print("SHAP EXPLAINABILITY ANALYSIS")
    print("=" * 60)

    # ---------------------------------------
    # Load Dataset
    # ---------------------------------------
    print("\nLoading dataset...")
    df = load_data()

    # ---------------------------------------
    # Data Preprocessing
    # ---------------------------------------
    print("Cleaning dataset...")
    df = clean_data(df)

    # ---------------------------------------
    # Feature Engineering
    # ---------------------------------------
    print("Performing feature engineering...")
    df = engineer_features(df)

    # ---------------------------------------
    # Prepare Train/Test Data
    # ---------------------------------------
    print("Preparing train/test datasets...")

    (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
        feature_columns,
    ) = prepare_training_data(df)

    X_train = pd.DataFrame(
        X_train,
        columns=feature_columns
    )

    X_test = pd.DataFrame(
        X_test,
        columns=feature_columns
    )

    print(f"\nTraining Samples : {len(X_train)}")
    print(f"Testing Samples  : {len(X_test)}")
    print(f"Number of Features : {len(feature_columns)}")

    # ---------------------------------------
    # Generate SHAP Report
    # ---------------------------------------
    print("\nGenerating SHAP report...")

    generate_shap_report(
        X_train,
        X_test,
        feature_columns
    )

    print("\n" + "=" * 60)
    print("SHAP REPORT GENERATED SUCCESSFULLY")
    print("=" * 60)

    print("\nGenerated Files:")

    print("reports/csv/shap_feature_importance.csv")

    print("reports/figures/shap_summary.png")

    print("reports/figures/shap_bar.png")

    print("reports/figures/shap_waterfall.png")

    print("reports/figures/shap_dependence_<top_feature>.png")


if __name__ == "__main__":
    main()