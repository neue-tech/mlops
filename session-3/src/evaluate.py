import mlflow
from sklearn.metrics import classification_report
from src.utils import save_json

def log_classification_report(y_true, y_pred, out_path="artifacts/classification_report.json"):
    report = classification_report(y_true, y_pred, output_dict=True)
    save_json(report, out_path)
    mlflow.log_artifact(out_path)
    return report
