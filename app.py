import streamlit as st
import pandas as pd
import joblib
import numpy as np

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# -------------------------
# Custom CSS (Styling)
# -------------------------
st.markdown(
    """
    <style>
    .main {
        background-color: #0E1117;
    }

    h1 {
        color: #FF4B4B;
        text-align: center;
        font-size: 42px;
    }

    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        font-size: 18px;
        border-radius: 10px;
        height: 3em;
        width: 100%;
    }

    .stButton>button:hover {
        background-color: #ff1f1f;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------
# Load Model
# -------------------------
model = joblib.load("fraud_model.pkl")

# -------------------------
# Title Section
# -------------------------
st.title("💳 Credit Card Fraud Detection System")
st.caption("AI-Based Random Forest + SMOTE Model")

st.success("Model Loaded Successfully ✅")

st.markdown("### 📊 Enter Transaction Details")

# -------------------------
# Inputs
# -------------------------
time = st.number_input("Time", value=0.0)
amount = st.number_input("Amount", value=0.0)

v = []
cols = st.columns(3)

for i in range(1, 29):
    with cols[i % 3]:
        v.append(st.number_input(f"V{i}", value=0.0))

# -------------------------
# Prediction
# -------------------------
if st.button("🚨 Predict Fraud"):

    data = np.array([[time] + v + [amount]])

    prediction = model.predict(data)

    st.markdown("### 🔍 Result")

    if prediction[0] == 1:
        st.error("🚨 Fraud Transaction Detected")
        st.warning("High Risk Transaction")
    else:
        st.success("✅ Legitimate Transaction")
        st.info("Low Risk Transaction")