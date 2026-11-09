from src.data_loader import load_data
from src.visualization import (
    dataset_overview,
    correlation_heatmap,
    price_distribution
)

df = load_data()

dataset_overview(df)

correlation_heatmap(df)

price_distribution(df)