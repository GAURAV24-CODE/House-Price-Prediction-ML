import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset before model training.

    Steps:
    1. Remove unnecessary columns
    2. Handle missing values
    3. Convert date column
    """

    # Create a copy
    df = df.copy()

    # -----------------------------------
    # Drop unnecessary columns
    # -----------------------------------
    df.drop(columns=["Unnamed: 0", "id"], inplace=True)

    # -----------------------------------
    # Fill missing numerical values
    # -----------------------------------
    df["bedrooms"] = df["bedrooms"].fillna(df["bedrooms"].median())
    df["bathrooms"] = df["bathrooms"].fillna(df["bathrooms"].median())

    # -----------------------------------
    # Convert date to datetime
    # -----------------------------------
    df["date"] = pd.to_datetime(df["date"])

    return df