import pandas as pd


def select_features(
    importance_df: pd.DataFrame,
    threshold: float = 0.01
):
    """
    Select features with importance above the threshold.
    """

    selected = importance_df[
        importance_df["Importance"] >= threshold
    ]

    selected_features = selected["Feature"].tolist()

    print(f"\nSelected {len(selected_features)} features")

    return selected_features