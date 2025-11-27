import pandas as pd

class DataIngestion:
    def __init__(self, data_path):
        self.data_path = data_path

    def load_data(self):
        df = pd.read_csv(self.data_path)
        print(f"[DataIngestion] Loaded: {self.data_path}, shape={df.shape}")
        return df
