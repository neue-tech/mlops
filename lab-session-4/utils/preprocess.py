# utils/preprocess.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple

def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame, numeric_cols: list) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])
    return X_train_scaled, X_test_scaled, scaler
