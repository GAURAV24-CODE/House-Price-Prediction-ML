"""Feature engineering shared by model training and inference."""

import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create model features using each property's sale date as reference."""
    df = df.copy()

    # Derive sale date fields once, so training and prediction use the same logic.
    if "date" in df.columns:
        sale_dates = pd.to_datetime(df["date"], errors="coerce")
        if "sale_year" not in df.columns:
            df["sale_year"] = sale_dates.dt.year
        if "sale_month" not in df.columns:
            df["sale_month"] = sale_dates.dt.month
        df = df.drop(columns=["date"])

    if "sale_year" not in df.columns:
        raise ValueError("A 'sale_year' column or a valid 'date' column is required.")

    sale_year = pd.to_numeric(df["sale_year"], errors="coerce")
    built_year = pd.to_numeric(df["yr_built"], errors="coerce")
    renovated_year = pd.to_numeric(df["yr_renovated"], errors="coerce")

    df["house_age"] = (sale_year - built_year).clip(lower=0)
    df["is_renovated"] = (renovated_year > 0).astype(int)
    df["renovation_age"] = (sale_year - renovated_year).where(
        renovated_year > 0, 0
    ).clip(lower=0)
    df["bath_bed_ratio"] = (
        pd.to_numeric(df["bathrooms"], errors="coerce")
        / pd.to_numeric(df["bedrooms"], errors="coerce").replace(0, 1)
    )

    return df
