# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 23:21:42 2026

@author: yogen
"""

import pickle
import streamlit as st
import numpy as np
import pandas as pd
import os

# ======================================
# Page Configuration
# ======================================
st.set_page_config(
    page_title="Aadhaar Early Warning System",
    page_icon="⚠️",
    layout="centered"
)

# ======================================
# Base Directory
# ======================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ======================================
# Custom CSS
# ======================================
st.markdown("""
<style>
body { background-color: #0e1117; color: #ffffff; }
h1, h2, h3 { color: #00e5ff; }
.metric-box {
    background-color: #1e222a;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
}
.risk { color: #ff5252; font-weight: bold; }
.safe { color: #00c853; font-weight: bold; }
.footer {
    text-align: center;
    color: #9e9e9e;
    font-size: 13px;
    margin-top: 30px;
}
</style>
""", unsafe_allow_html=True)

# ======================================
# Title
# ======================================
st.markdown("""
# 🚨 Aadhaar Enrolment Early-Warning System  
### AI-Driven Detection of Suspicious Enrolment Patterns  
*Hackathon Ready*
""")

st.markdown("---")

# ======================================
# Load Models EXACTLY as in GitHub
# ======================================
@st.cache_resource
def load_models():
    model_50k = pickle.load(open(
        os.path.join(BASE_DIR, "FIRST_AADHAR_50K_ENROLMENT (1).pkl"), "rb"
    ))
    model_100k = pickle.load(open(
        os.path.join(BASE_DIR, "AADHAR_100K_ENROLMENT.pkl"), "rb"
    ))
    model_150k = pickle.load(open(
        os.path.join(BASE_DIR, "AADHAR_150K_ENROLMENT.pkl"), "rb"
    ))
    scaler = pickle.load(open(
        os.path.join(BASE_DIR, "scaler.sav"), "rb"
    ))
    return model_50k, model_100k, model_150k, scaler

model_50k, model_100k, model_150k, scaler = load_models()

# ======================================
# DEBUG (Optional – can remove later)
# ======================================
st.write("📂 Files in app directory:")
st.write(os.listdir(BASE_DIR))

st.write("🧪 Scaler expects features:", scaler.n_features_in_)

# ======================================
# Sidebar – Model Selection
# ======================================
st.sidebar.header("🧠 Model Selection")

model_choice = st.sidebar.radio(
    "Choose Enrolment Range",
    (
        "FIRST Aadhaar (≤ 50K)",
        "AADHAAR Enrolment (≤ 100K)",
        "AADHAAR Enrolment (≤ 150K)"
    )
)

if model_choice == "FIRST Aadhaar (≤ 50K)":
    model = model_50k
elif model_choice == "AADHAAR Enrolment (≤ 100K)":
    model = model_100k
else:
    model = model_150k

# ======================================
# User Inputs
# ======================================
st.header("🧾 User Input Details")

age = st.number_input("Enter Age", 0, 120, 30)
pincode_total = st.number_input("Pincode Total Enrolments", min_value=0)
district_avg = st.number_input("District Average Enrolments", min_value=0.0)
state_avg = st.number_input("State Average Enrolments", min_value=0.0)

total_enrolment = pincode_total  # 5th feature

# ======================================
# Prediction
# ======================================
if st.button("🔍 Predict Enrolment Risk"):

    input_data = np.array([[
        age,
        pincode_total,
        district_avg,
        state_avg,
        total_enrolment
    ]])

    scaled_data = scaler.transform(input_data)

    prediction = model.predict(scaled_data)[0]
    probability = model.predict_proba(scaled_data)[0][1]

    st.markdown("## 📊 Prediction Result")

    if prediction == 1:
        st.error("🚨 **SUSPICIOUS ENROLMENT DETECTED**")
        st.markdown(
            f"<div class='risk'>Risk Probability: {probability:.2f}</div>",
            unsafe_allow_html=True
        )
    else:
        st.success("✅ **NORMAL ENROLMENT**")
        st.markdown(
            f"<div class='safe'>Risk Probability: {probability:.2f}</div>",
            unsafe_allow_html=True
        )

    st.progress(int(probability * 100))
    st.write(f"Confidence Level: **{probability*100:.1f}%**")

# ======================================
# Footer
# ======================================
st.markdown("""
<div class="footer">
Built for National Hackathon 🚀<br>
Early Detection • Transparency • AI Governance
</div>
""", unsafe_allow_html=True)
