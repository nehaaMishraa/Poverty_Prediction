# 🌍 Multidimensional Poverty Prediction Web Application

This project provides an interactive web interface for a machine learning model that predicts the Multidimensional Poverty Headcount Ratio based on various deprivation indicators. The backend uses an XGBoost Regressor model trained on historical data, while the frontend is built with Streamlit, allowing users to input characteristics and receive real-time predictions.

## ✨ Features


* **Interactive Input:** Users can adjust six key deprivation rates (Education enrollment, Education attainment, Electricity, Sanitation, Drinking water, and overall Deprivation rate) using sliders in a user-friendly sidebar.

* **Real-time Prediction:** The trained XGBoost model provides an instant prediction of the Multidimensional Poverty Headcount Ratio (%) based on the inputs.

* **Clear Visualization:** The predicted poverty ratio is displayed prominently and visualized using a simple gauge-like bar chart for easy interpretation.

* **Model Reusability:** The machine learning model is trained once and saved, then loaded efficiently by the web application using Streamlit's caching mechanisms.

* **Pure Python:** The entire application, from data preprocessing and model training to the interactive web interface, is built purely in Python.

## 🚀 Technologies Used


* **Python 3.x**

* **Pandas:** For data manipulation and handling.

* **NumPy:** For numerical operations.

* **Scikit-learn:** For machine learning utilities (e.g., `train_test_split`, `GridSearchCV`).

* **XGBoost:** The powerful gradient boosting library used for the regression model (`XGBRegressor`).

* **Streamlit:** For building the interactive web application.

* **Joblib:** For saving and loading the trained machine learning model.

* **Matplotlib & Seaborn:** For creating data visualizations.

## 🛠️ Setup Instructions


Follow these steps to get the project up and running on your local machine.

### Prerequisites


* Python 3.8+ installed on your system.

* `pip` (Python package installer).

### 1. Clone the Repository (or download files)
* git clone "paste the link"
  <br>
 * cd "File name"

### Otherwise, ensure you have the following files in the same directory:

* python.py (your model training script)

* app.py (the Streamlit web application script)

* MPM-Historical-Data-AM2024-Final.xlsx (your dataset)

* poverty_prediction_model.pkl (will be generated in the next step)

### 2. Install Dependencies
 pip install pandas numpy scikit-learn joblib matplotlib seaborn xgboost streamlit openpyxl

### 3. Run the training script:
 python python.py
### 🚀 Usage:
---
* Run the Streamlit app:
  <br>
  streamlit run app.py
 
 ### ⚠️ Disclaimer
 ---
This machine learning model is for illustrative and informational purposes only and is based on the provided historical data and selected features. 
<br>It should not be used as the sole basis for critical decisions related to multidimensional poverty assessment or policy-making. 
<br> Real-world poverty analysis is complex and requires comprehensive data, expert interpretation, and consideration of many socio-economic factors not necessarily captured by this model.

