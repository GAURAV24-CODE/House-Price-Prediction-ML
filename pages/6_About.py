import streamlit as st

st.set_page_config(
    page_title="About Project",
    page_icon="📘",
    layout="wide"
)

st.title("📘 About the Project")

st.markdown("""
## 🏠 House Price Prediction using Machine Learning

This project predicts residential house prices using the **King County House Sales Dataset**.
It demonstrates an end-to-end Machine Learning workflow, from data preprocessing to model deployment through an interactive Streamlit dashboard.
""")

st.divider()

# ============================================================
# Dataset
# ============================================================

st.header("📂 Dataset")

st.markdown("""
- **Dataset:** King County House Sales Dataset
- **Target Variable:** Price
- **Number of Samples:** ~21,000
- **Features:** 24 engineered features
- **Task:** Regression
""")

st.divider()

# ============================================================
# Project Workflow
# ============================================================

st.header("⚙️ Machine Learning Workflow")

st.markdown("""
1. Data Collection
2. Data Cleaning
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Feature Selection
6. Model Training
7. Hyperparameter Tuning
8. Cross Validation
9. Model Evaluation
10. SHAP Explainability
11. Production Model Selection
12. Streamlit Deployment
""")

st.divider()

# ============================================================
# Models
# ============================================================

st.header("🤖 Machine Learning Models")

st.markdown("""
The following regression models were implemented and compared:

- Linear Regression
- Ridge Regression
- Lasso Regression
- ElasticNet Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor (Production Model)
""")

st.divider()

# ============================================================
# Performance
# ============================================================

st.header("📈 Final Model Performance")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("R² Score", "0.881")

with col2:
    st.metric("RMSE", "$133,998")

with col3:
    st.metric("Cross Validation", "~0.90")

st.info(
    "The XGBoost model was selected as the production model after benchmarking, hyperparameter tuning, and 5-fold cross-validation."
)

st.divider()

# ============================================================
# Explainability
# ============================================================

st.header("🧠 Explainable AI")

st.markdown("""
Model predictions are interpreted using **SHAP (SHapley Additive exPlanations)**.

The dashboard includes:

- SHAP Summary Plot
- Global Feature Importance
- SHAP Dependence Plot
- SHAP Waterfall Plot
""")

st.divider()

# ============================================================
# Technologies
# ============================================================

st.header("🛠️ Technologies Used")

tech1, tech2, tech3 = st.columns(3)

with tech1:
    st.markdown("""
### Programming
- Python
- Pandas
- NumPy
""")

with tech2:
    st.markdown("""
### Machine Learning
- Scikit-learn
- XGBoost
- SHAP
""")

with tech3:
    st.markdown("""
### Visualization
- Matplotlib
- Streamlit
- Joblib
""")

st.divider()

# ============================================================
# Key Features
# ============================================================

st.header("⭐ Key Features")

st.markdown("""
- Interactive Streamlit Dashboard
- End-to-End ML Pipeline
- Automated Feature Engineering
- Model Comparison Dashboard
- Hyperparameter Tuning
- Cross Validation
- SHAP Explainability
- Real-time House Price Prediction
- Production-ready Project Structure
""")

st.divider()

# ============================================================
# Author
# ============================================================

st.header("👨‍💻 Author")

st.markdown("""
**Developed as an end-to-end Machine Learning project demonstrating:**

- Data Analysis
- Machine Learning
- Feature Engineering
- Model Evaluation
- Explainable AI (XAI)
- Streamlit Deployment
""")

st.success("✅ End-to-End Machine Learning Project Successfully Completed")