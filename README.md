## 🌾 Crop Yield Predictor & Recommendation System

An end-to-end, professional-grade machine learning project that predicts agricultural crop yields and recommends the optimal crop to plant based on environmental, soil, and farming data.

This project features a robust data engineering pipeline, comparative model evaluation, and an interactive web dashboard powered by Streamlit. It also integrates Explainable AI (SHAP) to provide transparent, feature-level breakdowns of every prediction.

## 📊 Dataset

- **Source:** [Agriculture Crop Yield Dataset on Kaggle](https://www.kaggle.com/datasets/samuelotiattakorah/agriculture-crop-yield)
- Contains regional, environmental, and agricultural features along with actual crop yields
- Used for both training and evaluation of the model

## 🚀 Key Features

**Automated Machine Learning Pipeline**: Evaluates and compares multiple state-of-the-art algorithms (Random Forest, XGBoost, Gradient Boosting) to select the optimal predictive model.

**Explainable AI (XAI)**: Utilizes SHAP (SHapley Additive exPlanations) waterfall charts to visually explain how each factor (e.g., Rainfall, Fertilizer, Soil Type) positively or negatively impacts the predicted yield.

**Robust Data Engineering**: Implements ColumnTransformer and OneHotEncoder for seamless categorical encoding, alongside memory optimization techniques (float32 casting) to handle large datasets (1,000,000+ rows) efficiently without RAM exhaustion.

**Interactive Web Application**: A user-friendly Streamlit dashboard for real-time predictions and crop recommendations.

## 🛠️ Technology Stack

**Language**: Python

**Machine Learning**: Scikit-Learn, XGBoost

**Explainable AI**: SHAP

**Data Processing & EDA**: Pandas, NumPy, Matplotlib, Seaborn

**Deployment & UI**: Streamlit, Joblib

## 📁 Project Structure

Crop_Yield_Predictor/
│

├── crop_yield.csv                 # Raw agricultural dataset (1M rows)

├── crop_yield_model.ipynb         # Model training, EDA, and evaluation notebook

├── crop_yield_app.py              # Streamlit dashboard application

├── requirements.txt               # Python dependencies

│

├── best_crop_yield_model.pkl      # Serialized XGBoost/Winning model

├── preprocessor.pkl               # Serialized ColumnTransformer (OneHotEncoding)

└── shap_explainer.pkl             # Serialized SHAP TreeExplainer


## 📊 Exploratory Data Analysis (EDA) Insights

During the data analysis phase, several critical drivers of crop yield were identified:

Rainfall & Water: Rainfall (mm) showed a massive positive correlation with crop yield, heavily dominating the predictive weighting alongside Irrigation usage.

Fertilizer: Farms utilizing fertilizer showed a significant statistical increase in average yield (5.40 tons/ha vs 3.89 tons/ha without).

Data Cleaning: Physically impossible anomalies (e.g., negative yield values) were programmatically identified and removed prior to pipeline execution.

## 💻 Installation & Setup

To run this project locally on your machine, follow these steps:

Step - 1: Clone this repository.

Step - 2: Install the required dependencies:
          pip install -r requirements.txt
          
Step - 3: Train the Model & Generate Artifacts

Step - 4: Launch the Streamlit Web App:
          streamlit run crop_yield_app.py


## 🔮 Future Scope

Integration of live weather APIs to pull real-time rainfall and temperature data based on the user's geographical location.

Implementation of advanced hyperparameter tuning (e.g., RandomizedSearchCV with K-Fold Cross Validation) on larger cloud-compute instances.
