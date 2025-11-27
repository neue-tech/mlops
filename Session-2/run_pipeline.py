import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
import sys
import os
sys.path.append("D:/ml-lab/mlops-hands-on")

from src.data_ingestion import DataIngestion
from src.data_preprocess import DataPreprocess
from src.train import TrainModel

from sklearn.model_selection import train_test_split


def main():

    print("\n========== MLOps Pipeline Started ==========\n")

    # 1️⃣ Load Data
    ingestion = DataIngestion(data_path="data/iris.csv")
    df = ingestion.load_data()

    # Split X, y
    X = df.drop("target", axis=1)
    y = df["target"]


    # 2️⃣ Preprocess
    pre = DataPreprocess()
    X_scaled = pre.scale_features(X)

    # Save the scaler for reproducibility
    pre.save_scaler("artifacts/scaler.pkl")

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    print("[Pipeline] Data ingestion + preprocessing completed.")

    # 3️⃣ Train Models
    trainer = TrainModel(
        config_path="experiments/model_configs.json",
        artifact_path="artifacts"
    )

    trainer.train_all(X_train, y_train, X_test, y_test)

    print("\n========== MLOps Pipeline Completed Successfully ==========\n")


if __name__ == "__main__":
    main()
