import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Model Comparison",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Machine Learning Model Comparison")

st.markdown("""
This page compares the performance of all machine learning models evaluated during the project.
The objective was to identify the best-performing model for deployment based on multiple evaluation metrics.
""")

FIGURES = Path("reports") / "figures"

st.divider()

# ==========================================================
# R² Comparison
# ==========================================================

st.header("📈 R² Score Comparison")

r2 = FIGURES / "r2_comparison.png"

if r2.exists():
    st.image(r2, use_container_width=True)
else:
    st.warning("r2_comparison.png not found")

st.markdown("""
**Interpretation**

- Higher R² indicates better predictive performance.
- XGBoost achieved the highest R² score and was selected as the production model.
""")

st.divider()

# ==========================================================
# RMSE Comparison
# ==========================================================

st.header("📉 RMSE Comparison")

rmse = FIGURES / "rmse_comparison.png"

if rmse.exists():
    st.image(rmse, use_container_width=True)
else:
    st.warning("rmse_comparison.png not found")

st.markdown("""
**Interpretation**

- RMSE measures prediction error in dollars.
- Lower RMSE indicates better prediction accuracy.
""")

st.divider()

# ==========================================================
# MAE Comparison
# ==========================================================

st.header("📊 MAE Comparison")

mae = FIGURES / "mae_comparison.png"

if mae.exists():
    st.image(mae, use_container_width=True)
else:
    st.warning("mae_comparison.png not found")

st.markdown("""
**Interpretation**

- MAE measures the average absolute prediction error.
- Lower values indicate more accurate predictions.
""")

st.divider()

# ==========================================================
# Cross Validation
# ==========================================================

st.header("🔁 Cross Validation")

cv = FIGURES / "cross_validation_r2.png"

if cv.exists():
    st.image(cv, use_container_width=True)
else:
    st.warning("cross_validation_r2.png not found")

st.info("""
The production model was validated using 5-fold cross-validation.
The average R² score remained close to 0.90, demonstrating strong generalization performance.
""")

st.divider()

# ==========================================================
# Training Time
# ==========================================================

st.header("⏱ Training Time")

train = FIGURES / "training_time.png"

if train.exists():
    st.image(train, use_container_width=True)
else:
    st.warning("training_time.png not found")

st.markdown("""
Training time compares the computational cost of each model.
More complex models generally require additional training time.
""")

st.divider()

# ==========================================================
# Prediction Time
# ==========================================================

st.header("⚡ Prediction Time")

pred = FIGURES / "prediction_time.png"

if pred.exists():
    st.image(pred, use_container_width=True)
else:
    st.warning("prediction_time.png not found")

st.markdown("""
Prediction time measures how quickly a trained model generates predictions.
Fast inference is important for deployment in real-world applications.
""")

st.divider()

# ==========================================================
# Production Model
# ==========================================================

st.header("🏆 Production Model")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Selected Model", "XGBoost")

with col2:
    st.metric("Test R²", "0.881")

with col3:
    st.metric("Cross Validation", "~0.90")

st.success("""
XGBoost was selected as the production model because it achieved the best balance between predictive accuracy, generalization, and robustness after hyperparameter tuning and 5-fold cross-validation.
""")