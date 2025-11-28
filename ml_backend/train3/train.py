import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, root_mean_squared_error

# --- LOAD & PREPARE DATA ---
df = pd.read_csv("dataset_combined_2_5yr.csv", index_col=0, parse_dates=True)

# select only ROI columns
roi_cols = [c for c in df.columns if "ROI" in c]
data = df[roi_cols].copy()

# replace inf/nan
data = data.replace([np.inf, -np.inf], np.nan).ffill().bfill()

# log returns for numerical stability
data = np.log1p(data)

# smooth ROI (5-day rolling mean)
data = data.rolling(5).mean().dropna()

# --- NORMALIZE ---
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(data)
scaled = np.nan_to_num(scaled)

# --- CREATE SEQUENCES ---
def create_sequences(data, seq_len=60):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i-seq_len:i])
        y.append(data[i])
    return np.array(X), np.array(y)

SEQ_LEN = 60
X, y = create_sequences(scaled, seq_len=SEQ_LEN)

# --- TRAIN TEST SPLIT ---
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# --- MODEL ---
def build_hybrid_model(input_shape, output_dim):
    inp = layers.Input(shape=input_shape)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.2))(inp)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.2))(x)
    
    # Attention
    attention = layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    x = layers.Add()([x, attention])
    x = layers.LayerNormalization()(x)
    
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(output_dim, activation='linear')(x)
    
    model = models.Model(inputs=inp, outputs=out)
    return model

model = build_hybrid_model((SEQ_LEN, X.shape[2]), y.shape[1])

opt = optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4)
model.compile(optimizer=opt, loss='mse', metrics=['mae'])

# --- CALLBACKS ---
cb = [
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5),
    callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    callbacks.ModelCheckpoint("models/hybrid_lstm_attention.keras", save_best_only=True)
]

# --- TRAIN ---
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=120,
    batch_size=32,
    callbacks=cb,
    verbose=1
)

# --- EVALUATE ---
pred = model.predict(X_test)
mae = mean_absolute_error(y_test, pred)
rmse = root_mean_squared_error(y_test, pred)
r2 = r2_score(y_test, pred)

print(f"\n✅ Final Evaluation:\nMAE: {mae:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.3f}")
