# app/ui_streamlit.py
import os
import json
import streamlit as st
import pandas as pd
from PIL import Image
import requests

st.set_page_config(page_title="Oncology MLOps Dashboard", layout="wide")

ARTIFACT_DIR = "../artifacts"
DRIFT_DIR = os.path.join(ARTIFACT_DIR, "drift")

st.title("Oncology Dashboard")

tabs = st.tabs(["Overview / Metrics", "Explainability (SHAP)", "Drift", "Inference"])

with tabs[0]:
    st.header("Model Metrics (from artifacts / mlflow exported files)")
    # Try to load classification report
    report_path = os.path.join(ARTIFACT_DIR, "classification_report.json")
    if os.path.exists(report_path):
        report = json.load(open(report_path))
        st.subheader("Classification Report (test set)")
        df = pd.DataFrame(report).T
        st.dataframe(df)
    else:
        st.info("classification_report.json not found in artifacts/ — run pipeline first.")

    st.subheader("Confusion matrix")
    cm_path = os.path.join(ARTIFACT_DIR, "best_confusion.png")
    if os.path.exists(cm_path):
        st.image(cm_path, caption="Confusion matrix", use_container_width=True)
    else:
        st.info("Confusion image not found.")

    st.subheader("Quick metrics")
    # Metrics file or mlflow read could be added; for now approximate from class report
    if os.path.exists(report_path):
        precision = report.get("1", {}).get("precision")
        recall = report.get("1", {}).get("recall")
        f1 = report.get("1", {}).get("f1-score")
        col1, col2, col3 = st.columns(3)
        col1.metric("Precision (positive)", f"{precision:.3f}" if precision else "N/A")
        col2.metric("Recall / Sensitivity", f"{recall:.3f}" if recall else "N/A")
        col3.metric("F1 (positive)", f"{f1:.3f}" if f1 else "N/A")
    else:
        st.write("Run pipeline to populate metrics.")

with tabs[1]:
    st.header("Explainability (SHAP)")
    summary = os.path.join(ARTIFACT_DIR, "shap_summary.png")
    beeswarm = os.path.join(ARTIFACT_DIR, "shap_beeswarm.png")
    if os.path.exists(summary):
        st.image(summary, caption="SHAP summary", use_container_width=True)
    else:
        st.info("SHAP summary not found — run pipeline to produce SHAP artifacts.")
    if os.path.exists(beeswarm):
        st.image(beeswarm, caption="SHAP beeswarm", use_container_width=True)

with tabs[2]:
    st.header("Drift")
    drift_json = os.path.join(DRIFT_DIR, "drift_summary.json")
    drift_top = os.path.join(DRIFT_DIR, "top_drift.csv")
    if os.path.exists(drift_json):
        data = json.load(open(drift_json))
        st.subheader("Drift metrics (psi / ks)")
        # Show top numeric columns and their psi/ks
        results = data.get("results", {})
        df = pd.DataFrame([
            {"column": k, "psi": v.get("psi"), "ks_stat": v.get("ks_stat"), "ks_pvalue": v.get("ks_pvalue")}
            for k, v in results.items()
        ])
        st.dataframe(df.sort_values(by="psi", ascending=False).reset_index(drop=True))
    else:
        st.info("No drift summary found. Run monitoring/drift_check.py or run pipeline_enhanced.")

    st.subheader("Plots")
    if os.path.exists(DRIFT_DIR):
        imgs = [p for p in os.listdir(DRIFT_DIR) if p.endswith("_dist.png")]
        for im in imgs:
            st.image(os.path.join(DRIFT_DIR, im), caption=im)

with tabs[3]:
    st.header("Inference / Test predictions")
    st.write("Send a test request to your local model serving endpoint.")
    api_url = st.text_input("Model endpoint (POST)", value="http://127.0.0.1:8000/predict")
    mean_radius = st.number_input("mean_radius", value=14.0, step=0.1)
    mean_texture = st.number_input("mean_texture", value=20.0, step=0.1)
    mean_perimeter = st.number_input("mean_perimeter", value=90.0, step=0.1)
    mean_area = st.number_input("mean_area", value=600.0, step=0.1)
    if st.button("Predict"):
        payload = {
            "mean_radius": float(mean_radius),
            "mean_texture": float(mean_texture),
            "mean_perimeter": float(mean_perimeter),
            "mean_area": float(mean_area)
        }
        try:
            r = requests.post(api_url, json=payload, timeout=10)
            r.raise_for_status()
            st.success(r.json())
        except Exception as e:
            st.error(f"Request failed: {e}")
            st.info("If serving with mlflow serve, ensure route expects the same payload; or use fastapi app.")

st.sidebar.markdown("**Run order:**\n1. python run_pipeline.py\n2. mlflow ui (to inspect runs)\n3. python src/register_pyfunc.py ...\n4. serve model\n5. streamlit run app/ui_streamlit.py")
