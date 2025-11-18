# 04_clustering_anomaly.py
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import joblib

df = pd.read_csv("data_clustering.csv")

model = KMeans(n_clusters=2, random_state=42)
df["cluster"] = model.fit_predict(df[["cpu_usage"]])

plt.scatter(df["cpu_usage"], [0]*len(df), c=df["cluster"])
plt.title("CPU Usage Clustering")
plt.xlabel("cpu_usage")
plt.yticks([])
plt.show()

print("Cluster for CPU=60:", model.predict([[60]])[0])
joblib.dump(model, "model_cpu_kmeans.pkl")
