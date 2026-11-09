import matplotlib.pyplot as plt
import pandas as pd

from src.config import FIGURES_DIR


def generate_bar_chart(
    results_df: pd.DataFrame,
    metric: str,
    title: str,
    xlabel: str,
    filename: str,
    higher_is_better: bool,
    value_format: str,
    text_offset,
):
    """
    Generate a professional horizontal bar chart.
    """

    # Sorting
    sorted_df = results_df.sort_values(
        by=metric,
        ascending=not higher_is_better
    )

    # Best model
    if higher_is_better:
        best_model = sorted_df.iloc[-1]["Model"]
    else:
        best_model = sorted_df.iloc[0]["Model"]

    # Colors
    colors = [
        "green" if model == best_model else "steelblue"
        for model in sorted_df["Model"]
    ]

    plt.figure(figsize=(10, 6))

    bars = plt.barh(
        sorted_df["Model"],
        sorted_df[metric],
        color=colors
    )

    # Value labels
    for bar in bars:

        width = bar.get_width()

        plt.text(
            width + text_offset,
            bar.get_y() + bar.get_height()/2,
            value_format.format(width),
            va="center",
            fontsize=9
        )

    plt.xlabel(xlabel)

    plt.ylabel("Model")

    plt.title(title)

    plt.grid(
        axis="x",
        linestyle="--",
        alpha=0.5
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def generate_r2_chart(results_df):

    generate_bar_chart(
        results_df=results_df,
        metric="R2",
        title="Model Comparison Based on R² Score (Higher is Better)",
        xlabel="R² Score",
        filename="r2_comparison.png",
        higher_is_better=True,
        value_format="{:.3f}",
        text_offset=0.005,
    )

def generate_rmse_chart(results_df):

    generate_bar_chart(
        results_df=results_df,
        metric="RMSE",
        title="Model Comparison Based on RMSE (Lower is Better)",
        xlabel="RMSE (Prediction Error)",
        filename="rmse_comparison.png",
        higher_is_better=False,
        value_format="{:,.0f}",
        text_offset=2000,
    )

def generate_mae_chart(results_df):

    generate_bar_chart(
        results_df=results_df,
        metric="MAE",
        title="Model Comparison Based on MAE (Lower is Better)",
        xlabel="MAE",
        filename="mae_comparison.png",
        higher_is_better=False,
        value_format="{:,.0f}",
        text_offset=1500,
    )

def generate_training_time_chart(results_df):

    generate_bar_chart(
        results_df=results_df,
        metric="Training Time",
        title="Training Time Comparison",
        xlabel="Seconds",
        filename="training_time.png",
        higher_is_better=False,
        value_format="{:.2f}s",
        text_offset=0.1,
    )

def generate_prediction_time_chart(results_df):

    generate_bar_chart(
        results_df=results_df,
        metric="Prediction Time",
        title="Prediction Time Comparison",
        xlabel="Seconds",
        filename="prediction_time.png",
        higher_is_better=False,
        value_format="{:.4f}s",
        text_offset=0.0005,
    )

