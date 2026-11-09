import matplotlib.pyplot as plt
import pandas as pd


def dataset_overview(df: pd.DataFrame):
    """
    Print basic information about the dataset.
    """

    print("=" * 50)
    print("DATASET OVERVIEW")
    print("=" * 50)

    print(f"\nRows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    print("\nColumn Names:")
    print(df.columns.tolist())

    print("\nData Types:")
    print(df.dtypes)

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nSummary Statistics:")
    print(df.describe())

def correlation_heatmap(df):
    """
    Plot correlation heatmap.
    """

    plt.figure(figsize=(14, 10))

    corr = df.corr(numeric_only=True)

    plt.imshow(corr)

    plt.colorbar()

    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)

    plt.yticks(range(len(corr.columns)), corr.columns)

    plt.title("Correlation Heatmap")

    plt.tight_layout()

    plt.show()

def price_distribution(df):
    """
    Plot price distribution.
    """

    plt.figure(figsize=(8,5))

    plt.hist(df["price"], bins=40)

    plt.title("House Price Distribution")

    plt.xlabel("Price")

    plt.ylabel("Frequency")

    plt.show()