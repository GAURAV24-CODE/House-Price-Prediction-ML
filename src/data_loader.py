import pandas as pd

from src.config import DATASET_PATH


def load_data():
    """
    Load the King County house sales dataset.

    Returns
    -------
    pandas.DataFrame
        Loaded dataset.
    """
    df = pd.read_csv(DATASET_PATH)
    return df