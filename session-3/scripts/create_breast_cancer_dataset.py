# scripts/create_breast_cancer_dataset.py
import os
import pandas as pd
from sklearn.datasets import load_breast_cancer
import numpy as np

os.makedirs("data", exist_ok=True)

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
# normalize column names (no spaces)
df.columns = [c.replace(" ", "_") for c in df.columns]
df.to_csv("data/breast_cancer.csv", index=False)
print("Saved data/breast_cancer.csv")

# optional drifted file for simulation
df_drift = df.copy()
np.random.seed(42)
df_drift['mean_radius'] = df_drift['mean_radius'] + np.random.normal(0.3, 0.5, size=len(df_drift))
df_drift.to_csv("data/breast_cancer_drift.csv", index=False)
print("Saved data/breast_cancer_drift.csv")
