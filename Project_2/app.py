import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ===============================
# Load trained model
# ===============================
model_file = "diabetes_linear_model.pkl"
with open(model_file, "rb") as f:
    model = pickle.load(f)

# ===============================
# Streamlit App UI
# ===============================
st.set_page_config(page_title="Diabetes Prediction App", layout="centered")

st.title("🩺 Diabetes Prediction App")
st.write("Enter patient details to predict the likelihood of diabetes.")

# User inputs
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glucose", min_value=0, max_value=200, value=120)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=150, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
age = st.number_input("Age", min_value=10, max_value=100, value=30)

# ===============================
# Prediction
# ===============================
if st.button("🔍 Predict"):
    # Arrange input into correct shape
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, dpf, age]])
    
    prediction = model.predict(input_data)[0]
    prediction_class = 1 if prediction >= 0.5 else 0

    # Show result
    st.subheader("Prediction Result:")
    if prediction_class == 1:
        st.error(f"⚠️ The model predicts **Diabetes** (Score: {prediction:.4f})")
    else:
        st.success(f"✅ The model predicts **No Diabetes** (Score: {prediction:.4f})")
