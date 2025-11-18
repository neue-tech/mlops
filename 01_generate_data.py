# 01_generate_data.py
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression, make_blobs

# --- Classification data ---
X_clf, y_clf = make_classification(
    n_samples=300,
    n_features=8,
    n_informative=4,
    n_classes=2,
    random_state=42,
)
df_clf = pd.DataFrame(X_clf, columns=[f"f{i}" for i in range(X_clf.shape[1])])
df_clf["label"] = y_clf
df_clf.to_csv("data_classification.csv", index=False)

# --- Regression data ---
X_reg, y_reg = make_regression(
    n_samples=300,
    n_features=1,
    noise=15.0,
    random_state=42,
)
df_reg = pd.DataFrame({"day": X_reg.flatten(), "price": y_reg})
df_reg.to_csv("data_regression.csv", index=False)

# --- Clustering data ---
X_blob, _ = make_blobs(
    n_samples=300,
    centers=[[10], [50]],
    cluster_std=[3, 2],
    random_state=42,
)
df_blob = pd.DataFrame({"cpu_usage": X_blob.flatten()})
df_blob.to_csv("data_clustering.csv", index=False)

print("Synthetic datasets generated.")
