# utils/logger_mlflow.py
import mlflow
import mlflow.sklearn
from typing import Dict

def log_classification_run(run_name: str, model, metrics: Dict, params: Dict = None, artifact_paths: Dict = None):
    with mlflow.start_run(run_name=run_name):
        if params:
            for k, v in params.items():
                mlflow.log_param(k, v)
        for k, v in metrics.items():
            if k != "confusion_matrix":
                mlflow.log_metric(k, float(v) if v is not None else None)
        # Log confusion matrix as artifact (if provided path)
        if artifact_paths and "confusion_matrix" in artifact_paths:
            mlflow.log_artifact(artifact_paths["confusion_matrix"])
        mlflow.sklearn.log_model(model, "model")

def log_regression_run(run_name: str, model, metrics: Dict, params: Dict = None):
    with mlflow.start_run(run_name=run_name):
        if params:
            for k,v in params.items():
                mlflow.log_param(k, v)
        for k,v in metrics.items():
            mlflow.log_metric(k, float(v))
        mlflow.sklearn.log_model(model, "model")
