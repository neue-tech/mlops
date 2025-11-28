import os
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class DataPreprocess:
    def __init__(self, artifact_dir="artifacts"):
        self.artifact_dir = artifact_dir
        os.makedirs(self.artifact_dir, exist_ok=True)
        self.scaler = StandardScaler()

    def prepare(self, df, target_col="target", test_size=0.2, random_state=42):
        X = df.drop(columns=[target_col])
        y = df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=test_size,
                                                            random_state=random_state,
                                                            stratify=y)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, y_train, y_test

    def save_scaler(self, path=None):
        path = path or os.path.join(self.artifact_dir, "scaler.pkl")
        with open(path, "wb") as f:
            pickle.dump(self.scaler, f)
        print(f"[Preprocess] Saved scaler at {path}")
        return path
