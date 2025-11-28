# run_pipeline_enhanced.py
import os
import json
import pickle
import time
import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data_ingestion import DataIngestion
from src.data_validation import DataValidation
from src.feature_engineering import add_clinical_features as add_features
from src.data_preprocess import DataPreprocess
from src.train import TrainModel
from src.utils import save_confusion_matrix, save_json, compute_auc

# Additional imports for explainability & metrics
import shap
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

# call drift module (we provide an improved version)
from monitoring import drift_check

ARTIFACT_DIR = "artifacts"
MLFLOW_URI = "sqlite:///mlflow.db"
REGISTER_NAME = "oncology_breast_classifier"

def compute_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0

def compute_ppv(y_true, y_pred):
    # PPV is same as precision for positive class
    return precision_score(y_true, y_pred)

def compute_npv(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fn) if (tn + fn) > 0 else 0.0

def save_shap_artifacts(model, X_train, feature_names, out_dir=ARTIFACT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    # create a small background sample for explainer
    background = X_train[np.random.choice(X_train.shape[0], min(50, X_train.shape[0]), replace=False)]
    try:
        explainer = shap.Explainer(model, background, feature_names=feature_names)
        shap_values = explainer(X_train[:200])  # sample
    except Exception as e:
        # fallback to KernelExplainer for sklearn models (slower)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X_train[:100])
        # shap_values may be list (for multiclass). Wrap for saving plots.
    # Summary plot
    try:
        plt.figure(figsize=(8,6))
        shap.summary_plot(shap_values, features=X_train[:200], feature_names=feature_names, show=False)
        fp = os.path.join(out_dir, "shap_summary.png")
        plt.tight_layout()
        plt.savefig(fp)
        plt.close()
    except Exception as e:
        print("Warning saving shap summary:", e)
        fp = None
    # Beeswarm (if available)
    try:
        plt.figure(figsize=(8,6))
        shap.plots.beeswarm(shap_values, show=False)
        bp = os.path.join(out_dir, "shap_beeswarm.png")
        plt.tight_layout()
        plt.savefig(bp)
        plt.close()
    except Exception:
        bp = None
    # Save a basic HTML explanation (if shap has html support)
    try:
        html_path = os.path.join(out_dir, "shap_summary.html")
        shap_values_html = shap.plots._waterfall.waterfall_legacy(shap_values[0], max_display=15, show=False)  # may be experimental
        # If the above is not available or errors, skip HTML saving
        # (Keeping this tolerant across environments)
    except Exception:
        html_path = None
    return {"shap_summary": fp, "shap_beeswarm": bp, "shap_html": html_path}

def main(data_path="data/breast_cancer.csv"):
    print("=== Enhanced pipeline start ===")
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("oncology_breast_cancer_enhanced")

    # Ingest & validate
    df = DataIngestion(path=data_path).load()
    DataValidation(required_cols=list(df.columns)).validate(df)

    # Features
    df_fe = add_features(df)
    feature_cols = list(df_fe.drop(columns=["target"]).columns)

    # Preprocess
    pre = DataPreprocess(artifact_dir=ARTIFACT_DIR)
    X_train, X_test, y_train, y_test = pre.prepare(df_fe, target_col="target")
    scaler_path = pre.save_scaler()

    # Training (existing module)
    trainer = TrainModel(config_path="experiments/model_configs.json", artifact_dir=ARTIFACT_DIR)
    best = trainer.train_and_log(X_train, y_train, X_test, y_test, feature_names=feature_cols)
    print("Selected best:", best)

    # Load best local model (trainer returns local_path)
    best_local = best.get("local_path")
    if not best_local or not os.path.exists(best_local):
        print("Best local model not found, attempting to fetch from model_uri")
        # try to download via mlflow if model_uri present
        model_uri = best.get("model_uri")
        if model_uri:
            local_dir = os.path.join(ARTIFACT_DIR, "best_model_download")
            os.makedirs(local_dir, exist_ok=True)
            local_path = mlflow.artifacts.download_artifacts(model_uri)
            best_local = local_path
    if not best_local:
        raise RuntimeError("Could not locate best model for explainability")

    with open(best_local, "rb") as f:
        model = pickle.load(f)

    # Attach governance tags for the selected run (so audit shows reason)
    selected_run = best.get("run_id")
    if selected_run:
        client = mlflow.tracking.MlflowClient()
        client.set_tag(selected_run, "risk_category", "oncology_diagnosis")
        client.set_tag(selected_run, "regulation_required", "FDA_SaMD")
        client.set_tag(selected_run, "explainability", "SHAP")
        client.set_tag(selected_run, "data_version", "v1.0")
        client.set_tag(selected_run, "model_owner", "clinical_ai_team")

    # Now evaluate and log additional clinical metrics and artifacts
    preds = model.predict(X_test)
    try:
        probs = model.predict_proba(X_test)[:,1]
    except Exception:
        probs = None

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)         # sensitivity
    f1 = f1_score(y_test, preds)
    auc = compute_auc(y_test, probs) if probs is not None else None
    spec = compute_specificity(y_test, preds)
    ppv = compute_ppv(y_test, preds)
    npv = compute_npv(y_test, preds)

    # Save / log classification report
    from sklearn.metrics import classification_report
    report = classification_report(y_test, preds, output_dict=True)
    report_path = os.path.join(ARTIFACT_DIR, "classification_report.json")
    save_json(report, report_path)

    # Save confusion matrix image
    cm_path = os.path.join(ARTIFACT_DIR, "best_confusion.png")
    save_confusion_matrix(y_test, preds, labels=[0,1], out_path=cm_path)

    # Log metrics & artifacts to MLflow under a short 'postprocessing' run
    with mlflow.start_run(run_name="postprocessing", nested=True) as run:
        mlflow.log_metric("accuracy", float(acc))
        mlflow.log_metric("precision", float(prec))
        mlflow.log_metric("recall_sensitivity", float(rec))
        mlflow.log_metric("specificity", float(spec))
        mlflow.log_metric("ppv", float(ppv))
        mlflow.log_metric("npv", float(npv))
        if auc is not None:
            mlflow.log_metric("auc", float(auc))
        mlflow.log_metric("f1", float(f1))
        mlflow.log_artifact(report_path)
        mlflow.log_artifact(cm_path)

        # Save model size & training duration metadata (if run info available)
        try:
            model_size_mb = os.path.getsize(best_local) / (1024 * 1024)
            mlflow.log_metric("model_size_mb", float(model_size_mb))
        except Exception:
            pass

    # --- Explainability (SHAP) ---
    try:
        shap_out = save_shap_artifacts(model, X_train, feature_cols, out_dir=ARTIFACT_DIR)
        with mlflow.start_run(run_name="explainability", nested=True):
            for k, v in shap_out.items():
                if v and os.path.exists(v):
                    mlflow.log_artifact(v, artifact_path="shap")
    except Exception as e:
        print("SHAP generation failed:", e)

    # --- Drift check (call the monitoring module) ---
    try:
        print("Running drift check...")
        drift_report = drift_check.main(base_path="data/breast_cancer.csv",
                                        cur_path="data/breast_cancer_drift.csv",
                                        out_dir=os.path.join(ARTIFACT_DIR, "drift"))
        # drift_report is a dict with paths & metrics
        with mlflow.start_run(run_name="drift_summary", nested=True):
            for k, v in drift_report.get("metrics", {}).items():
                mlflow.log_metric(k, float(v))
            # log artifacts (plots / json)
            for p in drift_report.get("artifacts", []):
                if os.path.exists(p):
                    mlflow.log_artifact(p, artifact_path="drift")
    except Exception as e:
        print("Warning: drift check failed:", e)

    print("=== Enhanced pipeline done ===")

if __name__ == "__main__":
    main()
