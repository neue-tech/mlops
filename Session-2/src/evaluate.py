from sklearn.metrics import accuracy_score
import json
import pickle
import mlflow

class EvaluateModel:
    def evaluate_model(self, model_path, X_test, y_test):
        model = pickle.load(open(model_path, "rb"))
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        mlflow.log_metric("accuracy", acc)

        return acc
