import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt

#Loading the pkl files
model = joblib.load("best_crop_yield_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")
explainer = joblib.load("shap_explainer.pkl")

# Hardcoded classes extracted from the dataset
region_classes = ['East', 'North', 'South', 'West']
soil_classes = ['Chalky', 'Clay', 'Loam', 'Peaty', 'Sandy', 'Silt']
crop_classes = ['Barley', 'Cotton', 'Maize', 'Rice', 'Soybean', 'Wheat']
weather_classes = ['Cloudy', 'Rainy', 'Sunny']

# Title of the web page
st.title("🌾 Crop Yield Predictor & Crop Recommendation")

# Inputting the environmental data
st.header("Input Environmental & Farming Data")
region = st.selectbox("Region", region_classes)
soil = st.selectbox("Soil Type", soil_classes)
crop = st.selectbox("Crop", crop_classes)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, step=1.0)
temperature = st.number_input("Temperature (°C)", min_value=-10.0, step=0.1)
fertilizer = st.selectbox("Fertilizer Used", ["Yes", "No"])
irrigation = st.selectbox("Irrigation Used", ["Yes", "No"])
weather = st.selectbox("Weather Condition", weather_classes)
days = st.number_input("Days to Harvest", min_value=1, step=1)

# Function to create DataFrame matching the training data structure
def create_input_df(c):
    return pd.DataFrame([[
        region, soil, c, rainfall, temperature,
        fertilizer == "Yes", irrigation == "Yes", weather, days
    ]], columns=[
        "Region", "Soil_Type", "Crop", "Rainfall_mm", "Temperature_Celsius",
        "Fertilizer_Used", "Irrigation_Used", "Weather_Condition", "Days_to_Harvest"
    ])

# Predict crop yield
if st.button("Predict Yield for Selected Crop"):
    input_df = create_input_df(crop)
    processed_input = preprocessor.transform(input_df)
    prediction = model.predict(processed_input)[0]
    st.success(f"Estimated Yield: {prediction:.2f} tons per hectare")
    
    # SHAP Visualization
    st.subheader("Prediction Breakdown")
    st.write("See how each factor impacted your predicted yield:")
    shap_values = explainer(processed_input)
    shap_values.feature_names = list(preprocessor.get_feature_names_out())
    
    fig, ax = plt.subplots(figsize=(8, 4))
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)

# Recommend best crop
if st.button("Recommend Best Crop to Grow"):
    best_yield = -1
    best_crop = None

    for crop_option in crop_classes:
        temp_input = create_input_df(crop_option)
        processed_temp = preprocessor.transform(temp_input)
        yield_est = model.predict(processed_temp)[0]
        
        if yield_est > best_yield:
            best_yield = yield_est
            best_crop = crop_option

    st.info(f"🌱 Recommended Crop: **{best_crop}** with expected yield: **{best_yield:.2f} tons/hectare**")
