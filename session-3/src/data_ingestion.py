import pandas as pd
import os

class DataIngestion:
    def __init__(self, path="data/breast_cancer.csv"):
        self.path = path

    def load(self):
        if not os.path.exists(self.path):
            raise FileNotFoundError(self.path)
        df = pd.read_csv(self.path)
        print(f"[DataIngestion] Loaded {self.path} shape={df.shape}")
        return df
