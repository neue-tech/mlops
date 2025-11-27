import os
import json
import pickle
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


class TrainModel:
    def __init__(self, config_path="experiments/model_configs.json", artifact_path="artifacts"):
        self.config_path = config_path
        self.artifact_path = artifact_path

        os.makedirs(artifact_path, exist_ok=True)

        with open(config_path, "r") as f:
            self.configs = json.load(f)

        # 💾 Use stable backend
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("iris_exp")

    def get_model(self, model_type, params):
        if model_type == "logistic_regression":
            return LogisticRegression(**params)
        elif model_type == "random_forest":
            return RandomForestClassifier(**params)
        elif model_type == "svc":
            return SVC(**params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def train_all(self, X_train, y_train, X_test, y_test):
        for model_name, cfg in self.configs.items():

            model_type = cfg["model_type"]
            params = cfg["params"]

            with mlflow.start_run(run_name=model_name):

                # Build and train model
                model = self.get_model(model_type, params)
                model.fit(X_train, y_train)

                # Evaluate
                preds = model.predict(X_test)
                accuracy = accuracy_score(y_test, preds)

                # ✨ Log Params
                mlflow.log_param("model_type", model_type)
                for k, v in params.items():
                    mlflow.log_param(k, v)

                # ✨ Log Metrics
                mlflow.log_metric("accuracy", accuracy)

                # ✨ Save model to local artifacts
                model_path = os.path.join(self.artifact_path, f"{model_name}.pkl")
                with open(model_path, "wb") as f:
                    pickle.dump(model, f)

                # ✨ Add model signature + example for MLflow
                input_example = X_train[:1]

                mlflow.sklearn.log_model(
                    sk_model=model,
                    name=model_name,           # replaces deprecated artifact_path
                    input_example=input_example,
                )

                print(f"[Train] Trained {model_name} | accuracy={accuracy:.4f}")

        print("\n[Train] All models trained successfully.\n")
