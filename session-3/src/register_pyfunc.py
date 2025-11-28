import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import shutil
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
import pandas as pd
from src.pyfunc_wrapper import SklearnPyfunc



def register_pyfunc(
    sklearn_model_path="../artifacts/rf.pkl",
    scaler_path="../artifacts/scaler.pkl",
    feature_cols=None,
    register_name="oncology_breast_classifier"
):
    """
    Wrap a sklearn model + scaler as an MLflow PyFunc model,
    log it, and register it in the MLflow model registry.
    """

    # -----------------------------
    # MLflow Tracking Setup
    # -----------------------------
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("pyfunc_wrap")

    # Default features
    feature_cols = feature_cols or [
        "mean_radius",
        "mean_texture",
        "mean_perimeter",
        "mean_area"
    ]

    # -----------------------------
    # Prepare Artifact Folder
    # -----------------------------
    pyfunc_folder = "pyfunc_artifact"
    os.makedirs(pyfunc_folder, exist_ok=True)

    # Copy artifacts locally
    model_filename = os.path.basename(sklearn_model_path)
    scaler_filename = os.path.basename(scaler_path)

    model_dst = os.path.join(pyfunc_folder, model_filename)
    scaler_dst = os.path.join(pyfunc_folder, scaler_filename)

    shutil.copy2(sklearn_model_path, model_dst)
    shutil.copy2(scaler_path, scaler_dst)

    # -----------------------------
    # Create PyFunc Wrapper
    # -----------------------------
    pyfunc_model = SklearnPyfunc(
        model_path=model_dst,
        scaler_path=scaler_dst,
        feature_cols=feature_cols,
    )

    # -----------------------------
    # Create MLflow Signature
    # -----------------------------
    sample_input = pd.DataFrame([[0.1, 0.2, 0.3, 0.4]], columns=feature_cols)
    sample_output = pd.DataFrame({"prediction": [1]})
    signature = infer_signature(sample_input, sample_output)

    # -----------------------------
    # Log Model in MLflow
    # -----------------------------
    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model(
            artifact_path="pyfunc_model",
            python_model=pyfunc_model,
            artifacts={
                "sklearn_model": model_dst,
                "scaler": scaler_dst,
            },
            signature=signature
        )

        run_id = run.info.run_id
        model_uri = f"runs:/{run_id}/pyfunc_model"

    # -----------------------------
    # Register Model
    # -----------------------------
    client = MlflowClient()

    # Create registry entry if missing
    try:
        client.create_registered_model(register_name)
    except Exception:
        pass  # registered model already exists

    # Register new version
    mv = client.create_model_version(
        name=register_name,
        source=model_uri,
        run_id=run_id
    )

    # Promote to Production and archive old versions
    client.transition_model_version_stage(
        name=register_name,
        version=mv.version,
        stage="Production",
        archive_existing_versions=True
    )

    print(f"✔ Registered PyFunc model '{register_name}' version {mv.version}")
    print(f"✔ Model URI: {model_uri}")
