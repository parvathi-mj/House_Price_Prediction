import streamlit as st
import pickle
import numpy as np

# Page setup
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# Load the model
import joblib

@st.cache_resource
def load_model():
    return joblib.load('house_price_model.pkl')

model = load_model()

# App title
st.title("🏠 House Price Predictor")
st.write("Built with Python + Streamlit | By Parvathi M J")
st.divider()

# Input form
st.subheader("Enter House Details")

col1, col2 = st.columns(2)

with col1:
    bedrooms = st.slider("Bedrooms", 1, 10, 3)
    bathrooms = st.slider("Bathrooms", 1.0, 8.0, 2.0, step=0.25)
    sqft_living = st.number_input("Living Area (sqft)", 500, 10000, 2000)
    sqft_lot = st.number_input("Lot Size (sqft)", 500, 100000, 5000)
    floors = st.selectbox("Floors", [1.0, 1.5, 2.0, 2.5, 3.0])
    waterfront = st.selectbox("Waterfront?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    view = st.slider("View Quality (0-4)", 0, 4, 0)
    condition = st.slider("Condition (1-5)", 1, 5, 3)

with col2:
    grade = st.slider("Grade (1-13)", 1, 13, 7)
    sqft_above = st.number_input("Above Ground sqft", 500, 10000, 1500)
    sqft_basement = st.number_input("Basement sqft", 0, 5000, 500)
    yr_built = st.slider("Year Built", 1900, 2015, 1990)
    yr_renovated = st.number_input("Year Renovated (0 if never)", 0, 2015, 0)
    lat = st.number_input("Latitude", 47.0, 48.0, 47.5)
    long = st.number_input("Longitude", -122.5, -121.0, -122.0)
    sqft_living15 = st.number_input("Avg Neighbour Living sqft", 500, 6000, 1800)
    sqft_lot15 = st.number_input("Avg Neighbour Lot sqft", 500, 50000, 5000)

st.divider()

# Predict button
if st.button("🔮 Predict Price", type="primary", use_container_width=True):
    features = np.array([[
        bedrooms, bathrooms, sqft_living, sqft_lot,
        floors, waterfront, view, condition, grade,
        sqft_above, sqft_basement, yr_built, yr_renovated,
        lat, long, sqft_living15, sqft_lot15
    ]])
    
    prediction = model.predict(features)[0]
    
    st.success(f"### Estimated Price: ${prediction:,.0f}")
    st.info(f"Range: ${prediction*0.9:,.0f} – ${prediction*1.1:,.0f}")

st.divider()
st.caption("Model: Random Forest | Trained on 17,000+ King County house sales")
