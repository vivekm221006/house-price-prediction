import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Page config
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 California House Price Prediction")
st.markdown("### Random Forest Model with Uncertainty & Explainability")

# Sidebar inputs
st.sidebar.header("🔧 Input Features")

med_inc = st.sidebar.slider("Median Income ($10k)", 0.5, 15.0, 3.5, 0.1)
house_age = st.sidebar.slider("House Age (years)", 1, 52, 15)
ave_rooms = st.sidebar.slider("Average Rooms", 1.0, 20.0, 5.0, 0.5)
ave_bedrms = st.sidebar.slider("Average Bedrooms", 0.5, 5.0, 1.0, 0.1)
population = st.sidebar.slider("Population", 3, 35000, 1500, 100)
ave_occup = st.sidebar.slider("Average Occupancy", 0.5, 20.0, 3.0, 0.5)
latitude = st.sidebar.slider("Latitude", 32.0, 42.0, 37.0, 0.1)
longitude = st.sidebar.slider("Longitude", -125.0, -114.0, -119.0, 0.1)

st.sidebar.markdown("---")

if st.sidebar.button("🔮 Predict House Price", use_container_width=True):

    model_path = "outputs/models/random_forest.pkl"

    if not os.path.exists(model_path):
        st.error("❌ Model not found. Train the model first.")
        st.stop()

    model = joblib.load(model_path)

    # Prepare input
    input_df = pd.DataFrame({
        "MedInc": [med_inc],
        "HouseAge": [house_age],
        "AveRooms": [ave_rooms],
        "AveBedrms": [ave_bedrms],
        "Population": [population],
        "AveOccup": [ave_occup],
        "Latitude": [latitude],
        "Longitude": [longitude],
    })

    input_df["RoomsPerBedroom"] = input_df["AveRooms"] / (input_df["AveBedrms"] + 0.01)
    input_df["RoomsPerPerson"] = input_df["AveRooms"] / (input_df["AveOccup"] + 0.01)
    input_df["HouseholdsPerPopulation"] = input_df["AveOccup"] / (input_df["Population"] + 0.01)

    # 📈 CONFIDENCE INTERVALS
    tree_preds = np.array([
        tree.predict(input_df)[0]
        for tree in model.estimators_
    ])

    mean_pred = tree_preds.mean()
    std_pred = tree_preds.std()

    price_mean = mean_pred * 100000
    price_low = (mean_pred - 1.96 * std_pred) * 100000
    price_high = (mean_pred + 1.96 * std_pred) * 100000

    st.success("✅ Prediction Complete")

    st.markdown(
        f"""
        <h1 style='text-align:center;color:#2ecc71'>
            ${price_mean:,.0f}
        </h1>
        <p style='text-align:center'>
            95% Confidence Interval:
            ${price_low:,.0f} – ${price_high:,.0f}
        </p>
        """,
        unsafe_allow_html=True
    )

# Model metrics
st.markdown("---")
st.subheader("📊 Model Performance")

if os.path.exists("outputs/model_metrics.csv"):
    st.dataframe(pd.read_csv("outputs/model_metrics.csv"), use_container_width=True)

# Footer
st.markdown("---")
st.caption(
    "⚠️ California housing data was capped at $500k. "
    "Model retrained after removing capped values to prevent saturation."
)
