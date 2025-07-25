import streamlit as st
import pandas as pd
import numpy as np
import joblib # Used to load the saved model
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor # Import XGBRegressor as it's part of the model structure

# --- 1. Streamlit Page Configuration ---
# Sets up the basic page settings like title and layout.
st.set_page_config(
    page_title="Poverty Prediction App",
    layout="centered", # Can be "wide" for more space
    initial_sidebar_state="expanded" # Sidebar starts open
)

# --- Explanation of CSS Styling ---
# This block uses st.markdown with unsafe_allow_html=True to inject custom CSS.
# It improves the visual appeal of the app, making buttons look nicer,
# adding shadows, and setting background colors.
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6; /* Light gray background for the main content area */
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); /* Subtle shadow for depth */
    }
    .stButton>button {
        background-color: #4CAF50; /* Green button */
        color: white;
        font-weight: bold;
        padding: 10px 20px;
        border-radius: 8px; /* Rounded corners for buttons */
        border: none;
        cursor: pointer;
        transition: background-color 0.3s ease; /* Smooth hover effect */
        width: 100%; /* Make button full width of its container */
    }
    .stButton>button:hover {
        background-color: #45a049; /* Darker green on hover */
    }
    .stNumberInput, .stSelectbox, .stSlider {
        margin-bottom: 15px; /* Spacing between input widgets */
    }
    h1, h2, h3 {
        color: #2c3e50; /* Darker text for headers */
    }
    .stAlert {
        border-radius: 8px; /* Rounded corners for info/warning/error messages */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- 2. Load Your Machine Learning Model --- 
@st.cache_resource
def load_poverty_model():
    try:
        # Load the best_model saved from your python.py script
        model = joblib.load('poverty_prediction_model.pkl')
        return model
    except FileNotFoundError:
        st.error("Error: 'poverty_prediction_model.pkl' not found.")
        st.write("Please ensure you have run your `python.py` script after adding the `joblib.dump` line to save the model, and that the `.pkl` file is in the same directory as `app.py`.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while loading the model: {e}")
        st.write("Ensure all necessary libraries (like xgboost) are installed and compatible.")
        return None

# Load the model when the app starts
model = load_poverty_model()

# --- 3. Define Features for User Input ---
# This dictionary defines the input widgets for each feature your model expects.
# The keys are the display names, and the values are dictionaries specifying
# the widget type, min/max values, and default.
# These ranges (0-100) are chosen assuming the inputs are percentages.
features_info = {
    "Deprivation rate (share of population)": {"min": 0.0, "max": 100.0, "default": 10.0, "step": 0.1},
    "Education enrollment deprivation rate (share of population)": {"min": 0.0, "max": 100.0, "default": 5.0, "step": 0.1},
    "Education attainment deprivation rate (share of population)": {"min": 0.0, "max": 100.0, "default": 15.0, "step": 0.1},
    "Electricity deprivation rate (share of population)": {"min": 0.0, "max": 100.0, "default": 20.0, "step": 0.1},
    "Sanitation deprivation rate (share of population)": {"min": 0.0, "max": 100.0, "default": 25.0, "step": 0.1},
    "Drinking water deprivation rate (share of population)": {"min": 0.0, "max": 100.0, "default": 10.0, "step": 0.1},
}

# --- 4. Streamlit App Layout and User Interaction ---

st.title("🌍 Multidimensional Poverty Prediction")
st.write("Adjust the deprivation rates below to see the predicted poverty headcount ratio.")

# Create input widgets in a sidebar for better organization
st.sidebar.header("Input Features")
user_inputs = {}
# Loop through the features_info dictionary to create a number input for each feature
for feature_name, info in features_info.items():
    user_inputs[feature_name] = st.sidebar.number_input(
        f"**{feature_name}**", # Display name for the input
        min_value=info["min"],
        max_value=info["max"],
        value=info["default"],
        step=info["step"],
        format="%.2f" # Format to 2 decimal places for percentages
    )

# Convert user inputs into a Pandas DataFrame
input_df = pd.DataFrame([user_inputs])
input_df = input_df[list(features_info.keys())] # Ensure correct column order

st.subheader("Current Input Values:")
st.dataframe(input_df.style.format("{:.2f}")) # Display the input DataFrame

# --- 5. Make Prediction and Display Results ---
if model is not None:
    if st.button("Predict Poverty Headcount Ratio"):
        try:
            # Perform prediction using the loaded model
            # The model predicts a continuous value (percentage)
            predicted_poverty_ratio = model.predict(input_df)[0]

            st.markdown("---")
            st.subheader("Prediction Result:")

            # Display the prediction clearly
            st.success(f"**Predicted Multidimensional Poverty Headcount Ratio:** {predicted_poverty_ratio:.2f}%")

            # --- Visualization of the Prediction ---
            # Create a simple gauge-like visualization for the predicted percentage
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.barh([0], [predicted_poverty_ratio], color='#4CAF50', height=0.5)
            ax.set_xlim(0, 100) # Poverty ratio is 0-100%
            ax.set_yticks([]) # Hide y-axis ticks
            ax.set_xticks(np.arange(0, 101, 20)) # Show ticks every 20%
            ax.set_xlabel("Multidimensional Poverty Headcount Ratio (%)")
            ax.set_title("Predicted Poverty Level")
            # Add text label for the predicted value
            ax.text(predicted_poverty_ratio, 0, f'{predicted_poverty_ratio:.2f}%',
                    va='center', ha='left' if predicted_poverty_ratio < 90 else 'right',
                    color='white' if predicted_poverty_ratio > 50 else 'black',
                    fontsize=12, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig) # Display the Matplotlib figure in Streamlit

            st.markdown("---")
            st.info("Disclaimer: This prediction is based on the provided input features and the trained machine learning model. It is for illustrative purposes and should not be used as the sole basis for critical decisions.")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.write("Please check your input values and ensure the model is compatible with the input format.")
else:
    st.warning("Model could not be loaded. Please check the console for errors and ensure your model file exists and is correctly named 'poverty_prediction_model.pkl'.")

st.sidebar.markdown("---")


