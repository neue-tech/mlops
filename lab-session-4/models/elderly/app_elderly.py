# models/elderly/app_elderly.py
import streamlit as st
import pandas as pd
import mlflow.sklearn

st.title("Elderly Care - Hospital Stay Prediction (Demo)")

st.markdown("Run `python train_elderly.py` first to create MLflow runs.")

run_input = st.text_input("MLflow model local path (example: mlruns/1/<run_id>/artifacts/model)", value="")
if run_input:
    try:
        model = mlflow.sklearn.load_model(run_input)
    except Exception as e:
        st.error(f"Could not load model: {e}")
        model = None
else:
    model = None

age = st.number_input("Age", 60, 100, 75)
bp = st.number_input("Blood Pressure", 80, 200, 140)
hr = st.number_input("Heart Rate", 50, 150, 80)
mob = st.slider("Mobility Score (1-8)", 1, 8, 4)
chronic = st.slider("Chronic Score (1-5)", 1, 5, 2)

if st.button("Predict") and model is not None:
    df = pd.DataFrame([[age, bp, hr, mob, chronic]], columns=["age","bp","heart_rate","mobility_score","chronic_score"])
    try:
        pred = model.predict(df)[0]
        st.success(f"Expected Hospital Stay: {pred:.2f} days")
    except Exception as e:
        st.error(f"Prediction error: {e}")
