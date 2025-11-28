import mlflow.pyfunc
import pandas as pd
import joblib
import numpy as np

class SklearnPyfunc(mlflow.pyfunc.PythonModel):

    def __init__(self, model_path, scaler_path, feature_cols):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.feature_cols = feature_cols

    def load_context(self, context):
        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        # Ensure we only use required features
        X = model_input[self.feature_cols]

        # Scale
        X_scaled = self.scaler.transform(X)

        # Predict
        preds = self.model.predict(X_scaled)

        # Return clean output for API/Streamlit
        return pd.DataFrame({"prediction": preds})
