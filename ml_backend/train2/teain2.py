import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import ta  # pip install ta

# ======== Load combined CSV ========
df = pd.read_csv("all_assets_6months.csv")
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values(['Symbol', 'Date'], inplace=True)

# ======== Feature Engineering ========
def add_features(group):
    group['returns'] = group['Close'].pct_change()
    group['volatility'] = group['returns'].rolling(7).std()
    group['ma_7'] = group['Close'].rolling(7).mean()
    group['ma_21'] = group['Close'].rolling(21).mean()
    group['rsi'] = ta.momentum.RSIIndicator(group['Close'], window=14).rsi()
    group['target'] = group['Close'].shift(-5) / group['Close'] - 1  # 5-day ROI
    return group.dropna()

df = df.groupby('Symbol', group_keys=False).apply(add_features).dropna()

# ======== Scaling ========
features = ['Close', 'Volume', 'returns', 'volatility', 'ma_7', 'ma_21', 'rsi']
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

# ======== Sequence preparation ========
SEQ_LEN = 60
X, y = [], []

for sym in df['Symbol'].unique():
    data = df[df['Symbol']==sym]
    for i in range(len(data) - SEQ_LEN):
        X.append(data[features].iloc[i:i+SEQ_LEN].values)
        y.append(data['target'].iloc[i+SEQ_LEN])

X, y = np.array(X), np.array(y)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

# ======== Model ========
model = Sequential([
    Bidirectional(LSTM(128, return_sequences=True, input_shape=(SEQ_LEN, len(features)))),
    Dropout(0.3),
    BatchNormalization(),

    Bidirectional(LSTM(64, return_sequences=False)),
    Dropout(0.3),

    Dense(64, activation='relu'),
    Dense(1)  # ROI %
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss='mse', metrics=['mae', 'mape'])

# ======== Training ========
early = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
reduce = tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-5)
ckpt = tf.keras.callbacks.ModelCheckpoint('best_model.keras', save_best_only=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=64,
    callbacks=[early, reduce, ckpt],
    verbose=1
)

# ======== Evaluation ========
val_loss, val_mae, val_mape = model.evaluate(X_val, y_val)
print(f"Validation MAE: {val_mae:.4f}, MAPE: {val_mape:.2f}%")

model.save("final_general_lstm.keras")


from sklearn.metrics import r2_score, mean_squared_error

preds = model.predict(X_val).flatten()
mae = np.mean(np.abs(preds - y_val))
rmse = np.sqrt(mean_squared_error(y_val, preds))
r2 = r2_score(y_val, preds)
direction_acc = np.mean(np.sign(preds) == np.sign(y_val))

print(f"\nFinal Evaluation:")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.3f}")
print(f"Directional Accuracy: {direction_acc*100:.2f}%")
