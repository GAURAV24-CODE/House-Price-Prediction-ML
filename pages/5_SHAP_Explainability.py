import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="SHAP Explainability",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 SHAP Explainability Dashboard")

st.markdown("""
Understand **why** the XGBoost model predicts a particular house price.

SHAP (SHapley Additive exPlanations) explains the contribution of each feature to the model's prediction.
""")

FIGURES = Path("reports") / "figures"

st.divider()

# ==========================================================
# SHAP Summary
# ==========================================================

st.header("📊 SHAP Summary")

summary = FIGURES / "shap_summary.png"

if summary.exists():
    st.image(summary, use_container_width=True)
else:
    st.warning("shap_summary.png not found")

st.divider()

# ==========================================================
# Feature Importance
# ==========================================================

st.header("⭐ Global Feature Importance")

bar = FIGURES / "shap_bar.png"

if bar.exists():
    st.image(bar, use_container_width=True)
else:
    st.warning("shap_bar.png not found")

st.divider()

# ==========================================================
# Dependence Plot
# ==========================================================

st.header("📍 Feature Dependence")

dependence = FIGURES / "shap_dependence_lat.png"

if dependence.exists():
    st.image(dependence, use_container_width=True)
else:
    st.warning("shap_dependence_lat.png not found")

st.divider()

# ==========================================================
# Waterfall Plot
# ==========================================================

st.header("💧 Individual Prediction Explanation")

waterfall = FIGURES / "shap_waterfall.png"

if waterfall.exists():
    st.image(waterfall, use_container_width=True)
else:
    st.warning("shap_waterfall.png not found")

st.divider()

st.success("SHAP visualizations loaded successfully.")