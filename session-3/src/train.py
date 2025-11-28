import os
import json
import pickle
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.utils import save_confusion_matrix, save_json, compute_auc

class TrainModel:
    def __init__(self, config_path="experiments/model_configs.json", artifact_dir="artifacts"):
        self.artifact_dir = artifact_dir
        os.makedirs(self.artifact_dir, exist_ok=True)
        with open(config_path, "r") as f:
            self.configs = json.load(f)
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("oncology_breast_cancer")
        self.client = MlflowClient()

    def _get_model(self, model_type, params):
        if model_type == "logistic_regression":
            return LogisticRegression(**params)
        if model_type == "random_forest":
            return RandomForestClassifier(**params)
        if model_type == "xgboost":
            if XGBClassifier is None:
                raise ImportError("please install xgboost")
            return XGBClassifier(**params)
        raise ValueError("Unknown model")

    def train_and_log(self, X_train, y_train, X_test, y_test, feature_names=None):
        best = {"metric": -1, "run_id": None, "name": None, "model_uri": None}
        for name, cfg in self.configs.items():
            model_type = cfg["model_type"]
            params = cfg.get("params", {})
            with mlflow.start_run(run_name=name) as run:
                model = self._get_model(model_type, params)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                try:
                    probs = model.predict_proba(X_test)[:,1]
                except Exception:
                    probs = None
                acc = accuracy_score(y_test, preds)
                prec = precision_score(y_test, preds)
                rec = recall_score(y_test, preds)
                f1 = f1_score(y_test, preds)
                auc = compute_auc(y_test, probs) if probs is not None else None
                # Log params & metrics
                mlflow.log_param("model_type", model_type)
                mlflow.log_params(params)
                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("precision", prec)
                mlflow.log_metric("recall", rec)
                mlflow.log_metric("f1", f1)
                if auc is not None:
                    mlflow.log_metric("auc", auc)
                # Save local model
                local_path = os.path.join(self.artifact_dir, f"{name}.pkl")
                with open(local_path, "wb") as f:
                    pickle.dump(model, f)
                # Log sklearn model to mlflow (with input_example)
                input_example = X_test[:1]
                mlflow.sklearn.log_model(sk_model=model, name=name, input_example=input_example)
                # Confusion matrix
                cm_path = os.path.join(self.artifact_dir, f"{name}_confusion.png")
                save_confusion_matrix(y_test, preds, labels=[0,1], out_path=cm_path)
                mlflow.log_artifact(cm_path)
                run_id = run.info.run_id
                model_uri = f"runs:/{run_id}/{name}"
                metric_for_selection = rec  # prefer recall for oncology (catch disease)
                if metric_for_selection > best["metric"]:
                    best = {"metric": metric_for_selection, "run_id": run_id, "name": name, "model_uri": model_uri, "local_path": local_path}
                print(f"[Train] {name} acc={acc:.3f} recall={rec:.3f} f1={f1:.3f} run_id={run_id}")
        # Register best model
        register_name = "oncology_breast_classifier"
        try:
            self.client.create_registered_model(register_name)
        except Exception:
            pass
        mv = self.client.create_model_version(name=register_name, source=best["model_uri"], run_id=best["run_id"])
        self.client.transition_model_version_stage(name=register_name, version=mv.version, stage="Production", archive_existing_versions=True)
        print(f"[Train] Registered {register_name} v{mv.version} as Production (selected by recall={best['metric']:.3f})")
        return best
