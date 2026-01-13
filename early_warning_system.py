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
# Base Directory (IMPORTANT)
# ======================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ======================================
# Custom CSS (UI Enhancement)
# ======================================
st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: #ffffff;
}
.main {
    padding: 2rem;
}
h1, h2, h3 {
    color: #00e5ff;
}
.metric-box {
    background-color: #1e222a;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0 0 10px rgba(0,229,255,0.25);
}
.risk {
    color: #ff5252;
    font-weight: bold;
    font-size: 18px;
}
.safe {
    color: #00c853;
    font-weight: bold;
    font-size: 18px;
}
.footer {
    text-align: center;
    color: #9e9e9e;
    font-size: 13px;
    margin-top: 30px;
}
</style>
""", unsafe_allow_html=True)

# ======================================
# Title Section
# ======================================
st.markdown("""
# 🚨 Aadhaar Enrolment Early-Warning System  
### AI-Driven Detection of Suspicious Enrolment Patterns  
*Hackathon Ready | Explainable AI | Governance Analytics*
""")

st.markdown("---")

# ======================================
# Load Models and Scaler (DEPLOYMENT SAFE)
# ======================================
@st.cache_resource
def load_models():
    model_50k = pickle.load(open(
        os.path.join(BASE_DIR, "models", "FIRST_AADHAR_50K_ENROLMENT.pkl"), "rb"
    ))
    model_100k = pickle.load(open(
        os.path.join(BASE_DIR, "models", "AADHAR_100K_ENROLMENT.pkl"), "rb"
    ))
    model_150k = pickle.load(open(
        os.path.join(BASE_DIR, "models", "AADHAR_150K_ENROLMENT.pkl"), "rb"
    ))
    scaler = pickle.load(open(
        os.path.join(BASE_DIR, "models", "scaler.sav"), "rb"
    ))
    return model_50k, model_100k, model_150k, scaler

model_50k, model_100k, model_150k, scaler = load_models()

# ======================================
# Load Location Data
# ======================================
location_df = pd.read_csv(
    os.path.join(BASE_DIR, "data", "location_stats.csv")
)

# ======================================
# Debug (Safe for Hackathon)
# ======================================
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

state = st.selectbox(
    "Select State",
    sorted(location_df['state'].unique())
)

district = st.selectbox(
    "Select District",
    sorted(location_df[location_df['state'] == state]['district'].unique())
)

pincode = st.selectbox(
    "Select Pincode",
    sorted(
        location_df[
            (location_df['state'] == state) &
            (location_df['district'] == district)
        ]['pincode'].unique()
    )
)

# ======================================
# KPI Cards
# ======================================
st.markdown("## 📌 Input Summary")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(
        f"<div class='metric-box'>👤<br><b>Age</b><br>{age}</div>",
        unsafe_allow_html=True
    )

with c2:
    st.markdown(
        f"<div class='metric-box'>📍<br><b>Pincode</b><br>{pincode}</div>",
        unsafe_allow_html=True
    )

with c3:
    st.markdown(
        f"<div class='metric-box'>🧠<br><b>Model</b><br>{model_choice}</div>",
        unsafe_allow_html=True
    )

st.markdown("---")

# ======================================
# Prediction Section
# ======================================
if st.button("🔍 Predict Enrolment Risk"):

    row = location_df[
        (location_df['state'] == state) &
        (location_df['district'] == district) &
        (location_df['pincode'] == pincode)
    ]

    if row.empty:
        st.error("❌ No data available for selected location")
    else:
        pincode_total = row['pincode_total'].values[0]
        district_avg = row['district_avg'].values[0]
        state_avg = row['state_avg'].values[0]

        # 5th Feature
        total_enrolment = pincode_total

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

        # ======================================
        # Risk Meter
        # ======================================
        st.markdown("### ⚠️ Risk Confidence Meter")
        st.progress(int(probability * 100))
        st.write(f"Confidence Level: **{probability*100:.1f}%**")

        # ======================================
        # Explainability
        # ======================================
        st.markdown("### 🧠 Explainability (Why this result?)")
        st.info(f"""
- 📌 **Pincode Enrolment:** {pincode_total}
- 📊 **District Average:** {district_avg}
- 🗺️ **State Average:** {state_avg}
- 👤 **Age Factor:** {age}

🔍 Significant deviation from district/state trends increases suspicion.
""")

# ======================================
# Footer
# ======================================
st.markdown("""
<div class="footer">
Built for National Hackathon 🚀<br>
Early Detection • Transparency • Data-Driven Governance
</div>
""", unsafe_allow_html=True)
