# models/elderly/train_elderly.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from utils.evaluator import regression_metrics
from utils.logger_mlflow import log_regression_run
import matplotlib.pyplot as plt

DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "elderly_care.csv"))

def generate_sample_csv(path):
    data = {
        "age":[72,65,80,77,69,84,73,68,90,76,70,82,67,75,89,71,78,66,74,79],
        "bp":[140,135,160,150,130,170,145,128,180,155,132,165,138,148,175,142,158,133,149,162],
        "heart_rate":[80,76,88,90,72,95,85,74,100,78,82,92,79,86,98,81,88,77,84,89],
        "mobility_score":[4,6,3,2,7,1,5,8,1,3,4,2,6,5,1,4,3,7,2,3],
        "chronic_score":[2,1,3,3,1,4,2,1,5,3,2,4,1,2,5,2,3,1,3,4],
        "LOS_days":[4,3,9,6,3,12,5,2,15,7,4,10,3,5,14,4,8,3,6,9]
    }
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Sample elderly CSV generated at {path}")

def main():
    if not os.path.exists(DATA_PATH):
        generate_sample_csv(DATA_PATH)

    df = pd.read_csv(DATA_PATH)
    X = df.drop("LOS_days", axis=1)
    y = df["LOS_days"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    preds_lr = lr.predict(X_test)
    metrics_lr = regression_metrics(y_test, preds_lr)
    log_regression_run("elderly_linear_regression", lr, metrics_lr, params={"model":"linear_regression"})
    print("Logged Linear Regression run to MLflow.")

    # Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    preds_rf = rf.predict(X_test)
    metrics_rf = regression_metrics(y_test, preds_rf)
    # save feature importance
    fi = pd.DataFrame({"feature": X.columns, "importance": rf.feature_importances_}).sort_values("importance", ascending=False)
    fi_path = os.path.join(os.path.dirname(__file__), "elderly_feature_importance.csv")
    fi.to_csv(fi_path, index=False)
    log_regression_run("elderly_rf_regressor", rf, metrics_rf, params={"model":"rf_regressor"})
    print("Logged Random Forest Regressor run to MLflow.")

if __name__ == "__main__":
    main()
