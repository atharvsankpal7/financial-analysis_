# train_hybrid.py
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, BatchNormalization, Bidirectional
import matplotlib.pyplot as plt

# === Load Data ===
df = pd.read_csv("dataset_cleaned.csv")
print(f"Loaded dataset shape: {df.shape}")

# Sort by date per ticker to ensure sequence order
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)

# === Prepare sequences ===
features = ['Close', 'Volume', 'Return', 'SMA_10', 'EMA_10', 'RSI', 'Volatility']
lookback = 60  # days

# Normalize across features (already roughly z-scored, this smooths further)
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

X, y = [], []

# Create time windows per ticker
for t in df['Ticker'].unique():
    dft = df[df['Ticker'] == t]
    values = dft[features].values
    target = dft['Return'].values
    for i in range(len(dft) - lookback):
        X.append(values[i:i+lookback])
        y.append(target[i+lookback])

X, y = np.array(X), np.array(y)
print(f"X: {X.shape}, y: {y.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# === Model ===
model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', input_shape=(lookback, len(features))),
    BatchNormalization(),
    Dropout(0.2),

    Bidirectional(LSTM(128, return_sequences=True)),
    Dropout(0.2),
    Bidirectional(LSTM(64)),
    Dense(64, activation='relu'),
    Dense(1)
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# === Train ===
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=80,
    batch_size=64,
    verbose=1
)

# === Predict & Evaluate ===
pred = model.predict(X_test)

mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

print("\n✅ Final Evaluation:")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")

# === Plot ===
plt.figure(figsize=(10,4))
plt.plot(y_test[:300], label="Actual", alpha=0.7)
plt.plot(pred[:300], label="Predicted", alpha=0.7)
plt.legend()
plt.title("Predicted vs Actual Returns (Test Sample)")
plt.tight_layout()
plt.savefig("prediction_plot.png", dpi=200)
plt.show()
