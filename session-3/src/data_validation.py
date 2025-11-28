class DataValidation:
    def __init__(self, required_cols=None):
        self.required_cols = required_cols

    def validate(self, df):
        if self.required_cols:
            missing = [c for c in self.required_cols if c not in df.columns]
            if missing:
                raise ValueError(f"Missing columns: {missing}")
        if df.isnull().sum().sum() > 0:
            print("[Validation] Warning: dataset has missing values")
        print("[Validation] OK")
        return True
