import os
import pandas as pd
from sklearn.datasets import load_iris

os.makedirs("data", exist_ok=True)

# Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"] = iris.target
df.to_csv("../data/iris.csv", index=False)

print("Created data/iris.csv")
