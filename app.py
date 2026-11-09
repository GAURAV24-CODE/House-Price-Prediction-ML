import streamlit as st

st.set_page_config(
    page_title="House Price Prediction Dashboard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🏠 House Price Prediction Dashboard")

st.markdown(
    """
    Welcome to the **House Price Prediction Dashboard**.

    This application demonstrates an end-to-end Machine Learning pipeline
    for predicting house prices using the King County House Sales dataset.

    ---
    """
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Best Model",
        value="XGBoost",
    )

with col2:
    st.metric(
        label="R² Score",
        value="0.881",
    )

with col3:
    st.metric(
        label="RMSE",
        value="133,998",
    )

st.markdown("---")

st.subheader("🚀 Project Highlights")

st.markdown(
    """
- End-to-End Machine Learning Pipeline
- Data Cleaning & Feature Engineering
- Exploratory Data Analysis (EDA)
- Model Comparison
- Hyperparameter Tuning
- Cross Validation
- SHAP Explainability
- Interactive Prediction Dashboard
"""
)

st.markdown("---")

st.subheader("📌 Dashboard Navigation")

st.info(
    """
Use the sidebar to navigate between:

- Dataset Explorer
- EDA Dashboard
- Model Comparison
- Price Prediction
- SHAP Explainability
- About Project
"""
)

st.markdown("---")

st.caption(
    "Developed by Sharan G • Streamlit • XGBoost • SHAP • Scikit-learn"
)