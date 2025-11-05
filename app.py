# app_key_features_fixed_layout.py
import streamlit as st
import pandas as pd
import joblib
from PIL import Image
import os

# Load model once
@st.cache_resource
def load_model():
    return joblib.load("breast_cancer_rf_keyfeatures.pkl")

model = load_model()

# Page Title
st.title("Breast Cancer Prediction")

# ---------------------
# Sidebar: Sliders
# ---------------------
st.sidebar.header("Adjust Key Features")

radius_mean = st.sidebar.slider("Radius Mean", 6.0, 30.0, 14.0)
texture_mean = st.sidebar.slider("Texture Mean", 9.0, 40.0, 20.0)
perimeter_mean = st.sidebar.slider("Perimeter Mean", 40.0, 190.0, 90.0)
area_mean = st.sidebar.slider("Area Mean", 140.0, 2500.0, 600.0)
smoothness_mean = st.sidebar.slider("Smoothness Mean", 0.05, 0.16, 0.1)

# Collect inputs
input_data = pd.DataFrame({
    'radius_mean': [radius_mean],
    'texture_mean': [texture_mean],
    'perimeter_mean': [perimeter_mean],
    'area_mean': [area_mean],
    'smoothness_mean': [smoothness_mean]
})

# ---------------------
# Main Page: Prediction
# ---------------------
# Use container to maintain layout
prediction_container = st.container()

with prediction_container:
    if st.button("Predict"):
        prediction = model.predict(input_data)[0]

        # Set label and background color
        if prediction == 0:
            label = "Positive (Cancerous)"
            bg_color = "#FFCCCC"  # light red
            img_file = "pos.jpeg"
        else:
            label = "Negative (Non-Cancerous)"
            bg_color = "#CCFFCC"  # light green
            img_file = "neg.jpeg"

        # Change page background dynamically
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-color: {bg_color};
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

        st.markdown(f"<h2 style='color:black'>{label}</h2>", unsafe_allow_html=True)
        st.info("Consult a healthcare professional for detailed guidance.")

        image_folder = "static"
        image_path = os.path.join(image_folder, img_file)
        if os.path.exists(image_path):
            image = Image.open(image_path)
            st.image(image, width=250)
        else:
            st.warning(f"Image {img_file} not found in {image_folder} folder.")
