"""
06_clustering_anomaly_weekly.py

Goal:
- Generate daily CPU usage for several weeks
- Simulate real DevOps patterns (weekday vs weekend)
- Add anomalies (spikes & drops)
- Detect anomalies using KMeans
- Plot weekly CPU usage
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# -----------------------------
# 1. Generate Synthetic Weekly CPU Data
# -----------------------------

weeks = 8
days = weeks * 7

day_index = np.arange(1, days + 1)

# CPU pattern:
# Weekdays: 40–70% usage
# Weekends: 20–40% usage
cpu_usage = []

for i in range(days):
    day_of_week = i % 7

    if day_of_week < 5:  # Monday–Friday
        usage = np.random.randint(40, 70) + np.random.normal(0, 3)
    else:  # Saturday–Sunday
        usage = np.random.randint(20, 40) + np.random.normal(0, 2)

    cpu_usage.append(usage)

cpu_usage = np.array(cpu_usage)

# Inject anomalies
cpu_usage[12] = 95   # Monday - extreme spike
cpu_usage[37] = 10   # Friday - extreme drop
cpu_usage[45] = 88   # Thursday - spike
cpu_usage[50] = 5    # Saturday - abnormal low

df = pd.DataFrame({
    "day": day_index,
    "week": (day_index - 1) // 7 + 1,
    "cpu_usage": cpu_usage
})

# -----------------------------
# 2. Clustering for Anomaly Detection
# -----------------------------

X = df["cpu_usage"].values.reshape(-1, 1)

kmeans = KMeans(n_clusters=3, random_state=42)
df["cluster"] = kmeans.fit_predict(X)

# Identify anomaly cluster
cluster_means = df.groupby("cluster")["cpu_usage"].mean().to_dict()
overall_mean = df["cpu_usage"].mean()

anomaly_cluster = max(
    cluster_means,
    key=lambda c: abs(cluster_means[c] - overall_mean)
)

df["is_anomaly"] = df["cluster"] == anomaly_cluster

# -----------------------------
# Display anomalies
# -----------------------------
print("\n=== CPU ANOMALIES (Daily | Weekly) ===")
print(df[df["is_anomaly"]][["day", "week", "cpu_usage"]])
print("=======================================\n")

# -----------------------------
# 3. Plot weekly pattern
# -----------------------------

plt.figure(figsize=(14, 7))

normal = df[~df["is_anomaly"]]
plt.scatter(normal["day"], normal["cpu_usage"], s=30, label="Normal Usage")

anomalies = df[df["is_anomaly"]]
plt.scatter(anomalies["day"], anomalies["cpu_usage"], s=50, color="red", label="Anomaly")

plt.title("CPU Usage Over Weeks (Clustering + Anomaly Detection)")
plt.xlabel("Day")
plt.ylabel("CPU Usage (%)")
plt.grid(True)
plt.legend()
plt.show()
