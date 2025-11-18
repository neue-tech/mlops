"""
05_clustering_anomaly.py
Goal:
- Generate synthetic time-series CPU usage data
- Use KMeans clustering to identify usage patterns
- Detect anomalous CPU usage (very high / very low)
- Plot results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# -----------------------------
# 1. Generate Synthetic CPU Data
# -----------------------------

# Simulate timestamps
time_steps = np.arange(1, 501)  # 500 time points

# Generate CPU usage pattern
cpu_normal = 40 + 10 * np.sin(time_steps / 20) + np.random.normal(0, 3, 500)

# Inject anomalies
cpu_normal[50] = 95     # High spike
cpu_normal[180] = 3     # Drop
cpu_normal[350] = 90    # High spike
cpu_normal[420] = 5     # Drop

df = pd.DataFrame({
    "time": time_steps,
    "cpu_usage": cpu_normal
})

# -----------------------------
# 2. Clustering using KMeans
# -----------------------------

# Reshape CPU usage into 2D as required by sklearn
X = df["cpu_usage"].values.reshape(-1, 1)

kmeans = KMeans(n_clusters=3, random_state=42)
df["cluster"] = kmeans.fit_predict(X)

# Compute cluster means
cluster_means = df.groupby("cluster")["cpu_usage"].mean().to_dict()

# Identify anomaly cluster = farthest cluster
# Anomaly cluster is the cluster with extreme mean CPU usage
anomaly_cluster = max(cluster_means, key=lambda c: abs(cluster_means[c] - df["cpu_usage"].mean()))

df["is_anomaly"] = df["cluster"] == anomaly_cluster

# -----------------------------
# 3. Display Anomalies
# -----------------------------
print("\n=== DETECTED CPU ANOMALIES ===")
print(df[df["is_anomaly"]][["time", "cpu_usage"]])
print("==============================\n")

# -----------------------------
# 4. Plot the Results
# -----------------------------

plt.figure(figsize=(12, 6))

# Plot normal points
normal = df[~df["is_anomaly"]]
plt.scatter(normal["time"], normal["cpu_usage"], label="Normal Usage", s=15)

# Plot anomalies
anomalies = df[df["is_anomaly"]]
plt.scatter(anomalies["time"], anomalies["cpu_usage"], color="red", label="Anomaly", s=40)

plt.title("CPU Usage Over Time (Clustering + Anomaly Detection)")
plt.xlabel("Time")
plt.ylabel("CPU Usage (%)")
plt.legend()
plt.grid(True)
plt.show()
