import streamlit as st
import pandas as pd

from src.data_loader import load_data

st.set_page_config(
    page_title="Dataset Explorer",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Dataset Explorer")

st.markdown(
    """
Explore the King County House Sales dataset used to train the machine learning models.
"""
)

# -------------------------------------------------
# Load Dataset
# -------------------------------------------------

df = load_data()

# -------------------------------------------------
# Basic Information
# -------------------------------------------------

st.subheader("Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Rows", f"{df.shape[0]:,}")

with col2:
    st.metric("Columns", df.shape[1])

with col3:
    st.metric("Missing Values", int(df.isna().sum().sum()))

st.divider()

# -------------------------------------------------
# Preview
# -------------------------------------------------

st.subheader("Dataset Preview")

rows = st.slider(
    "Number of rows",
    min_value=5,
    max_value=100,
    value=10,
)

st.dataframe(
    df.head(rows),
    use_container_width=True,
)

st.divider()

# -------------------------------------------------
# Data Types
# -------------------------------------------------

st.subheader("Column Information")

info = pd.DataFrame({
    "Column": df.columns,
    "Data Type": df.dtypes.astype(str),
    "Missing Values": df.isna().sum().values,
})

st.dataframe(
    info,
    use_container_width=True,
)

st.divider()

# -------------------------------------------------
# Statistical Summary
# -------------------------------------------------

st.subheader("Statistical Summary")

st.dataframe(
    df.describe(),
    use_container_width=True,
)

st.divider()

# -------------------------------------------------
# Missing Values
# -------------------------------------------------

st.subheader("Missing Values")

missing = pd.DataFrame({
    "Column": df.columns,
    "Missing Values": df.isna().sum(),
})

missing = missing.sort_values(
    by="Missing Values",
    ascending=False,
)

st.dataframe(
    missing,
    use_container_width=True,
)

st.success("Dataset loaded successfully.")