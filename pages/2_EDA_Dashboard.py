import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Model Performance",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Model Performance Dashboard")

st.markdown(
    """
This page summarizes the exploratory analysis, model evaluation,
cross-validation and diagnostic visualizations generated during
the machine learning workflow.
"""
)

FIGURES = Path("reports/figures")


def show_image(filename, title):
    path = FIGURES / filename

    if path.exists():
        st.subheader(title)
        st.image(str(path), use_container_width=True)
        st.divider()


# ===================================================
# Exploratory Analysis
# ===================================================

st.header("📊 Feature Analysis")

col1, col2 = st.columns(2)

with col1:
    if (FIGURES / "feature_importance.png").exists():
        st.image(
            str(FIGURES / "feature_importance.png"),
            caption="Feature Importance",
            use_container_width=True,
        )

with col2:
    if (FIGURES / "prediction_error_distribution.png").exists():
        st.image(
            str(FIGURES / "prediction_error_distribution.png"),
            caption="Prediction Error Distribution",
            use_container_width=True,
        )

st.divider()

# ===================================================
# Model Comparison
# ===================================================

st.header("🤖 Model Comparison")

col1, col2 = st.columns(2)

with col1:
    st.image(
        str(FIGURES / "r2_comparison.png"),
        caption="R² Comparison",
        use_container_width=True,
    )

with col2:
    st.image(
        str(FIGURES / "rmse_comparison.png"),
        caption="RMSE Comparison",
        use_container_width=True,
    )

st.image(
    str(FIGURES / "mae_comparison.png"),
    caption="MAE Comparison",
    use_container_width=True,
)

st.divider()

# ===================================================
# Cross Validation
# ===================================================

st.header("📉 Cross Validation")

col1, col2 = st.columns(2)

with col1:
    st.image(
        str(FIGURES / "cross_validation_r2.png"),
        caption="Cross Validation R²",
        use_container_width=True,
    )

with col2:
    st.image(
        str(FIGURES / "cross_validation_rmse.png"),
        caption="Cross Validation RMSE",
        use_container_width=True,
    )

st.divider()

# ===================================================
# Diagnostics
# ===================================================

st.header("🔍 Regression Diagnostics")

col1, col2 = st.columns(2)

with col1:
    st.image(
        str(FIGURES / "actual_vs_predicted.png"),
        caption="Actual vs Predicted",
        use_container_width=True,
    )

with col2:
    st.image(
        str(FIGURES / "residual_plot.png"),
        caption="Residual Plot",
        use_container_width=True,
    )

st.image(
    str(FIGURES / "residual_distribution.png"),
    caption="Residual Distribution",
    use_container_width=True,
)

st.divider()

# ===================================================
# Performance
# ===================================================

st.header("⚡ Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.image(
        str(FIGURES / "training_time.png"),
        caption="Training Time",
        use_container_width=True,
    )

with col2:
    st.image(
        str(FIGURES / "prediction_time.png"),
        caption="Prediction Time",
        use_container_width=True,
    )

st.success("All evaluation reports loaded successfully.")