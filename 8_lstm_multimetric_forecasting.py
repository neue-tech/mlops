"""
8_lstm_multimetric_forecasting.py
-----------------------------------
Multivariate LSTM Forecasting Example
Use case: Predict CPU, Memory, Disk usage for next time window
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# =======================================================
# 1. Load / Generate Synthetic Data
# =======================================================

def generate_system_metrics(n=5000):
    """
    Generate synthetic CPU, Memory, Disk metrics
    with realistic correlations and daily patterns.
    """
    time = pd.date_range("2024-01-01", periods=n, freq="T")  # per minute

    cpu = 40 + 10*np.sin(np.linspace(0, 50, n)) + np.random.normal(0, 2, n)
    memory = 60 + 5*np.sin(np.linspace(0, 20, n)) + np.random.normal(0, 1.5, n)
    disk = 50 + 8*np.sin(np.linspace(0, 30, n)) + np.random.normal(0, 1, n)

    df = pd.DataFrame({
        "timestamp": time,
        "cpu": cpu,
        "memory": memory,
        "disk": disk
    })

    return df

df = generate_system_metrics()
df.set_index("timestamp", inplace=True)
print(df.head())

# =======================================================
# 2. Scaling the Data
# =======================================================

scaler = MinMaxScaler()
scaled = scaler.fit_transform(df)

# =======================================================
# 3. Create Sliding Window Dataset
# =======================================================

def create_sequences(data, window=60):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(data[i+window])  # predict next step (multivariate)
    return np.array(X), np.array(y)

WINDOW = 60  # use last 60 minutes
X, y = create_sequences(scaled, WINDOW)

print("X shape:", X.shape)   # (samples, timesteps, features)
print("y shape:", y.shape)   # (samples, features)

# Split into train/test
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# =======================================================
# 4. Build LSTM Model
# =======================================================

model = Sequential([
    LSTM(64, return_sequences=False, input_shape=(WINDOW, 3)),
    Dense(32, activation="relu"),
    Dense(3)  # CPU, Memory, Disk
])

model.compile(optimizer="adam", loss="mse")
model.summary()

# =======================================================
# 5. Train Model
# =======================================================

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=32,
    verbose=1
)

# =======================================================
# 6. Make Predictions
# =======================================================

pred_scaled = model.predict(X_test)
pred = scaler.inverse_transform(pred_scaled)
actual = scaler.inverse_transform(y_test)

# =======================================================
# 7. Plot Predictions
# =======================================================

plt.figure(figsize=(14, 8))

plt.subplot(3, 1, 1)
plt.plot(actual[:200, 0], label="Actual CPU")
plt.plot(pred[:200, 0], label="Pred CPU")
plt.legend()
plt.title("CPU Forecasting")

plt.subplot(3, 1, 2)
plt.plot(actual[:200, 1], label="Actual Memory")
plt.plot(pred[:200, 1], label="Pred Memory")
plt.legend()
plt.title("Memory Forecasting")

plt.subplot(3, 1, 3)
plt.plot(actual[:200, 2], label="Actual Disk")
plt.plot(pred[:200, 2], label="Pred Disk")
plt.legend()
plt.title("Disk Forecasting")

plt.tight_layout()
plt.show()
