# Crop_Yield_Predictor
# 🌾 Crop Yield Prediction System

A machine learning-powered web application that predicts crop yield (in tons per hectare) based on environmental and agricultural inputs. It also recommends the best crop to grow under the given conditions. Built using **Python, scikit-learn, and Streamlit**.

---

## 🚀 Features

- 📈 **Crop Yield Prediction** using environmental factors (rainfall, temperature, etc.)
- 🌱 **Crop Recommendation** based on input conditions
- 🧠 Trained on 100,000+ rows of real-world agricultural data
- 💻 Interactive Streamlit web interface
- 🗂️ Model and encoders saved with `joblib`

---

## 🧠 Machine Learning Model

- Algorithm: `RandomForestRegressor`
- Training set: 100,000 stratified samples across crop types
- Input features:
  - Region
  - Soil Type
  - Crop
  - Rainfall (mm)
  - Temperature (°C)
  - Fertilizer Used (Yes/No)
  - Irrigation Used (Yes/No)
  - Weather Condition
  - Days to Harvest
- Output: `Yield_tons_per_hectare`

---

## 🖥️ Tech Stack

| Tool       | Purpose                  |
|------------|--------------------------|
| Python     | Core programming         |
| Pandas     | Data processing          |
| scikit-learn | ML model training      |
| Streamlit  | Web UI                   |
| joblib     | Model/encoder saving     |


