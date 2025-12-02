# models/oncology/train_oncology.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from utils.preprocess import scale_features
from utils.evaluator import classification_metrics
from utils.logger_mlflow import log_classification_run
import matplotlib.pyplot as plt
import seaborn as sns

# helper: generate small csv if none exists
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "oncology_data.csv")
DATA_PATH = os.path.abspath(DATA_PATH)

def generate_sample_csv(path):
    data = {
        "age":[45,61,52,37,50,68,55,70,42,59,65,72,58,49,62,80,74,47,53,66],
        "tumor_size_mm":[22,35,48,18,40,50,25,60,18,33,28,55,30,21,45,70,38,19,27,31],
        "lymph_nodes":[0,1,3,0,5,7,2,6,1,4,2,5,1,0,3,8,2,1,2,3],
        "stage":[1,2,3,1,2,3,2,4,1,3,2,4,2,1,3,4,2,1,2,3],
        "chemo_given":[0,1,1,0,0,1,0,1,1,0,1,1,0,0,1,1,1,0,0,1],
        "high_risk":[0,1,1,0,1,1,0,1,0,1,1,1,0,0,1,1,1,0,0,1]
    }
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Sample oncology CSV generated at {path}")

def plot_and_save_confusion(cm, outpath):
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def main():
    if not os.path.exists(DATA_PATH):
        generate_sample_csv(DATA_PATH)

    df = pd.read_csv(DATA_PATH)
    X = df.drop("high_risk", axis=1)
    y = df["high_risk"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    numeric_cols = ["age","tumor_size_mm","lymph_nodes","stage"]
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test, numeric_cols)

    # Logistic Regression
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train_scaled, y_train)
    preds_log = logreg.predict(X_test_scaled)
    probs_log = logreg.predict_proba(X_test_scaled)[:,1]
    metrics_log = classification_metrics(y_test, preds_log, probs_log)

    cm_path = os.path.join(os.path.dirname(__file__), "oncology_confusion_logreg.png")
    plot_and_save_confusion(metrics_log["confusion_matrix"], cm_path)

    log_classification_run("oncology_logreg", logreg, metrics_log, params={"model":"logistic_regression"},
                           artifact_paths={"confusion_matrix": cm_path})
    print("Logged Logistic Regression run to MLflow.")

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    preds_rf = rf.predict(X_test)
    probs_rf = rf.predict_proba(X_test)[:,1]
    metrics_rf = classification_metrics(y_test, preds_rf, probs_rf)

    cm_path_rf = os.path.join(os.path.dirname(__file__), "oncology_confusion_rf.png")
    plot_and_save_confusion(metrics_rf["confusion_matrix"], cm_path_rf)

    # Save feature importance CSV
    fi = pd.DataFrame({"feature": X.columns, "importance": rf.feature_importances_}).sort_values("importance", ascending=False)
    fi_path = os.path.join(os.path.dirname(__file__), "oncology_feature_importance.csv")
    fi.to_csv(fi_path, index=False)

    log_classification_run("oncology_rf", rf, metrics_rf, params={"model":"random_forest"},
                           artifact_paths={"confusion_matrix": cm_path_rf})
    print("Logged Random Forest run to MLflow.")

if __name__ == "__main__":
    main()
