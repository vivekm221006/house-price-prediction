import streamlit as st
import pandas as pd
import joblib
import os

# Page config
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# Title
st.title("🏠 California House Price Prediction")
st.markdown("### Machine Learning Model with 62% Accuracy")

# Sidebar
st.sidebar.header("🔧 Input Features")

# Input features
med_inc = st.sidebar.slider("Median Income ($10k)", 0.5, 15.0, 3.5, 0.1)
house_age = st.sidebar.slider("House Age (years)", 1, 52, 15)
ave_rooms = st.sidebar.slider("Average Rooms", 1.0, 20.0, 5.0, 0.5)
ave_bedrms = st.sidebar.slider("Average Bedrooms", 0.5, 5.0, 1.0, 0.1)
population = st.sidebar.slider("Population", 3, 35000, 1500, 100)
ave_occup = st.sidebar.slider("Average Occupancy", 0.5, 20.0, 3.0, 0.5)
latitude = st.sidebar.slider("Latitude", 32.0, 42.0, 37.0, 0.1)
longitude = st.sidebar.slider("Longitude", -125.0, -114.0, -119.0, 0.1)

st.sidebar.markdown("---")

# Predict button
if st.sidebar.button("🔮 Predict House Price", type="primary", use_container_width=True):

    model_path = "outputs/models/random_forest.pkl"

    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found at: {model_path}")
        st.stop()

    try:
        model = joblib.load(model_path)

        # Prepare input
        input_data = pd.DataFrame({
            "MedInc": [med_inc],
            "HouseAge": [house_age],
            "AveRooms": [ave_rooms],
            "AveBedrms": [ave_bedrms],
            "Population": [population],
            "AveOccup": [ave_occup],
            "Latitude": [latitude],
            "Longitude": [longitude],
        })

        # Feature engineering
        input_data["RoomsPerBedroom"] = input_data["AveRooms"] / (input_data["AveBedrms"] + 0.01)
        input_data["RoomsPerPerson"] = input_data["AveRooms"] / (input_data["AveOccup"] + 0.01)
        input_data["HouseholdsPerPopulation"] = input_data["AveOccup"] / (input_data["Population"] + 0.01)

        # Prediction
        prediction = model.predict(input_data)[0]
        price = prediction * 100000

        # Display result
        st.markdown("---")
        st.success("✅ Prediction Complete!")

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("### 💵 Predicted House Value")
            st.markdown(
                f"<h1 style='text-align: center; color: #2ecc71;'>${price:,.0f}</h1>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"<p style='text-align: center;'>Median House Value: {prediction:.2f} (in $100k)</p>",
                unsafe_allow_html=True
            )

        # Input summary
        st.markdown("---")
        st.subheader("📋 Input Summary")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Median Income", f"${int(med_inc * 10)}k")
            st.metric("House Age", f"{house_age} years")
        with c2:
            st.metric("Avg Rooms", f"{ave_rooms:.1f}")
            st.metric("Avg Bedrooms", f"{ave_bedrms:.1f}")
        with c3:
            st.metric("Population", f"{population:,}")
            st.metric("Avg Occupancy", f"{ave_occup:.1f}")
        with c4:
            st.metric("Latitude", f"{latitude:.1f}°")
            st.metric("Longitude", f"{longitude:.1f}°")

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
        st.info("Please try refreshing the page or contact support.")

# Model performance
st.markdown("---")
st.subheader("📊 Model Performance")

if os.path.exists("outputs/model_metrics.csv"):
    metrics_df = pd.read_csv("outputs/model_metrics.csv")
    st.dataframe(
        metrics_df.style.highlight_max(subset=["Test R²"], color="lightgreen"),
        use_container_width=True
    )
    st.success("✅ Best Model: **Random Forest** with **62.01% R² Score**")
else:
    st.info("Model metrics will be displayed here once available")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center;'>Built with ❤️ using Python, Scikit-learn & Streamlit</p>",
    unsafe_allow_html=True
)
