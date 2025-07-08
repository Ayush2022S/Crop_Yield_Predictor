
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the model and encoders
model = joblib.load("crop_yield_model.pkl")
region_encoder = joblib.load("Region_encoder.pkl")
soil_encoder = joblib.load("Soil_Type_encoder.pkl")
crop_encoder = joblib.load("Crop_encoder.pkl")
weather_encoder = joblib.load("Weather_Condition_encoder.pkl")

# Title of the web page
st.title("🌾 Crop Yield Predictor & Crop Recommendation")

# Input the environmental data
st.header("Input Environmental & Farming Data")
region = st.selectbox("Region", region_encoder.classes_)
soil = st.selectbox("Soil Type", soil_encoder.classes_)
crop = st.selectbox("Crop", crop_encoder.classes_)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, step=1.0)
temperature = st.number_input("Temperature (°C)", min_value=-10.0, step=0.1)
fertilizer = st.selectbox("Fertilizer Used", ["Yes", "No"])
irrigation = st.selectbox("Irrigation Used", ["Yes", "No"])
weather = st.selectbox("Weather Condition", weather_encoder.classes_)
days = st.number_input("Days to Harvest", min_value=1, step=1)

# Predict crop yield
if st.button("Predict Yield for Selected Crop"):
    input_data = pd.DataFrame([[
        region_encoder.transform([region])[0],
        soil_encoder.transform([soil])[0],
        crop_encoder.transform([crop])[0],
        rainfall,
        temperature,
        fertilizer == "Yes",
        irrigation == "Yes",
        weather_encoder.transform([weather])[0],
        days
    ]], columns=[
        "Region", "Soil_Type", "Crop", "Rainfall_mm", "Temperature_Celsius",
        "Fertilizer_Used", "Irrigation_Used", "Weather_Condition", "Days_to_Harvest"
    ])
    prediction = model.predict(input_data)[0]
    st.success(f"Estimated Yield: {prediction:.2f} tons per hectare")

# Recommend best crop
if st.button("Recommend Best Crop to Grow"):
    crops = crop_encoder.classes_
    best_yield = -1
    best_crop = None

    for crop_option in crops:
        temp_input = pd.DataFrame([[
            region_encoder.transform([region])[0],
            soil_encoder.transform([soil])[0],
            crop_encoder.transform([crop_option])[0],
            rainfall,
            temperature,
            fertilizer == "Yes",
            irrigation == "Yes",
            weather_encoder.transform([weather])[0],
            days
        ]], columns=[
            "Region", "Soil_Type", "Crop", "Rainfall_mm", "Temperature_Celsius",
            "Fertilizer_Used", "Irrigation_Used", "Weather_Condition", "Days_to_Harvest"
        ])
        yield_est = model.predict(temp_input)[0]
        if yield_est > best_yield:
            best_yield = yield_est
            best_crop = crop_option

    st.info(f"🌱 Recommended Crop: **{best_crop}** with expected yield: **{best_yield:.2f} tons/hectare**")
