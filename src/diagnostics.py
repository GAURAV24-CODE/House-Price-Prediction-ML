import matplotlib.pyplot as plt
import numpy as np

from src.config import FIGURES_DIR


def plot_actual_vs_predicted(y_true, y_pred):

    plt.figure(figsize=(7,7))

    plt.scatter(y_true, y_pred, alpha=0.6)

    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))

    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        "r--",
        linewidth=2
    )

    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs Predicted")
    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "actual_vs_predicted.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def plot_residuals(y_true, y_pred):

    residuals = y_true - y_pred

    plt.figure(figsize=(8,6))

    plt.scatter(
        y_pred,
        residuals,
        alpha=0.6
    )

    plt.axhline(
        y=0,
        color="red",
        linestyle="--"
    )

    plt.xlabel("Predicted Price")
    plt.ylabel("Residual")
    plt.title("Residual Plot")
    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "residual_plot.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def plot_residual_distribution(y_true, y_pred):

    residuals = y_true - y_pred

    plt.figure(figsize=(8,6))

    plt.hist(
        residuals,
        bins=30
    )

    plt.axvline(
        x=0,
        color="red",
        linestyle="--"
    )

    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.title("Residual Distribution")

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "residual_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def plot_prediction_error_distribution(y_true, y_pred):

    errors = np.abs(y_true - y_pred)

    plt.figure(figsize=(8,6))

    plt.hist(
        errors,
        bins=30
    )

    plt.xlabel("Absolute Error")
    plt.ylabel("Frequency")
    plt.title("Prediction Error Distribution")

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "prediction_error_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()