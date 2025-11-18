# 03_regression_stock.py
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import joblib

df = pd.read_csv("data_regression.csv")
df = df.sort_values("day")

X = df[["day"]]
y = df["price"]

model = LinearRegression()
model.fit(X, y)

preds = model.predict(X)

plt.scatter(X, y)
plt.plot(X, preds)
plt.title("Synthetic Stock Trend")
plt.xlabel("day")
plt.ylabel("price")
plt.show()

print("Example prediction for day=2.5:", model.predict([[2.5]])[0])
joblib.dump(model, "model_stock_regression.pkl")
