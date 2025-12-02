# models/oncology/app_oncology.py
import streamlit as st
import pandas as pd
import mlflow.sklearn
import os

st.title("Oncology Risk Predictor (Demo)")

# list runs/models in mlruns automatically is messy; ask user to provide run_path or search
st.markdown("**Note:** Run `python train_oncology.py` first to create MLflow runs.")

run_input = st.text_input("MLflow model local path (example: mlruns/0/<run_id>/artifacts/model)", value="")
if run_input:
    try:
        model = mlflow.sklearn.load_model(run_input)
    except Exception as e:
        st.error(f"Could not load model: {e}")
        model = None
else:
    model = None

age = st.number_input("Age", 18, 100, 60)
tumor = st.number_input("Tumor Size (mm)", 1, 200, 30)
nodes = st.number_input("Lymph Nodes", 0, 10, 1)
stage = st.slider("Stage", 1, 4, 2)
chemo = st.selectbox("Chemo Given", [0,1], index=1)

if st.button("Predict") and model is not None:
    df = pd.DataFrame([[age, tumor, nodes, stage, chemo]], columns=["age","tumor_size_mm","lymph_nodes","stage","chemo_given"])
    try:
        pred = model.predict(df)[0]
        st.success("High Risk" if pred==1 else "Low Risk")
    except Exception as e:
        st.error(f"Prediction error: {e}")
