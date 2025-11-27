import os
import pickle
from sklearn.preprocessing import StandardScaler

class DataPreprocess:
    def __init__(self):
        self.scaler = StandardScaler()

    def scale_features(self, X):
        """
        Fit and transform the features using StandardScaler
        """
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled

    def save_scaler(self, path="artifacts/scaler.pkl"):
        """
        Save the fitted scaler for future use
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.scaler, f)
        print(f"[Preprocess] Saved scaler at {path}")
