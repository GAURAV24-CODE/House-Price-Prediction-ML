import streamlit as st
from src.predictor import predict_house_price

st.set_page_config(
    page_title="Predict House Price",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 House Price Prediction")

st.write("Enter the property details below to estimate the house price.")

st.divider()

col1, col2 = st.columns(2)

with col1:
    bedrooms = st.number_input("Bedrooms", 1, 10, 3)
    bathrooms = st.number_input("Bathrooms", 1.0, 10.0, 2.0, step=0.5)
    sqft_living = st.number_input("Living Area (sqft)", 500, 15000, 2000)
    sqft_lot = st.number_input("Lot Size (sqft)", 500, 2000000, 5000)
    floors = st.selectbox("Floors", [1, 1.5, 2, 2.5, 3, 3.5])
    waterfront = st.selectbox("Waterfront", [0, 1])
    view = st.slider("View", 0, 4, 0)
    condition = st.slider("Condition", 1, 5, 3)
    grade = st.slider("Grade", 1, 13, 7)
    sqft_above = st.number_input("Above Ground Area", 0, 15000, 1500)
    sqft_basement = st.number_input("Basement Area", 0, 5000, 500)

with col2:
    yr_built = st.number_input("Year Built", 1900, 2025, 1995)
    yr_renovated = st.number_input("Year Renovated (0 if none)", 0, 2025, 0)
    zipcode = st.number_input("Zipcode", 98000, 98299, 98052)
    lat = st.number_input("Latitude", value=47.60, format="%.5f")
    long = st.number_input("Longitude", value=-122.20, format="%.5f")
    sqft_living15 = st.number_input("Neighbour Living Area", 500, 15000, 2000)
    sqft_lot15 = st.number_input("Neighbour Lot Area", 500, 2000000, 5000)
    sale_year = st.selectbox("Sale Year", [2014, 2015])
    sale_month = st.slider("Sale Month", 1, 12, 5)

st.divider()

if st.button("🔮 Predict House Price", use_container_width=True):

    features = {
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "sqft_living": sqft_living,
        "sqft_lot": sqft_lot,
        "floors": floors,
        "waterfront": waterfront,
        "view": view,
        "condition": condition,
        "grade": grade,
        "sqft_above": sqft_above,
        "sqft_basement": sqft_basement,
        "yr_built": yr_built,
        "yr_renovated": yr_renovated,
        "zipcode": zipcode,
        "lat": lat,
        "long": long,
        "sqft_living15": sqft_living15,
        "sqft_lot15": sqft_lot15,
        "sale_year": sale_year,
        "sale_month": sale_month,
    }

    try:
        prediction = predict_house_price(features)

        st.success("Prediction Completed Successfully!")

        st.metric(
            label="Estimated House Price",
            value=f"${prediction:,.2f}"
        )

    except Exception as e:
        st.error(str(e))