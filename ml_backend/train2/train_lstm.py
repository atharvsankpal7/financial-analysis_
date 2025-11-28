"""
train_lstm.py

Usage:
    python train_lstm.py

Config variables are near the top of the file. The script expects
`data/all_assets_6months.csv` created earlier.

Outputs:
- models/ (base_model.h5, finetuned_<SYMBOL>.h5)
- scalers/ (scaler_base.pkl, scaler_<SYMBOL>.pkl)
- logs/ (training history CSVs)
"""

import os
import random
import joblib
from datetime import datetime
from tqdm import tqdm
import numpy as np
import pandas as pd

# ML / DL
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras import backend as K

# Optional: ta library (technical indicators). If not present, install via pip.
try:
    import ta
except Exception:
    ta = None

# -----------------------
# CONFIG
# -----------------------
DATA_PATH = "all_assets_6months.csv"
OUTPUT_DIR = "models"
SCALER_DIR = "scalers"
LOG_DIR = "logs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SCALER_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

LOOKBACK = 60           # days used as input window
HORIZON = 1             # predict ROI after HORIZON days (1 => next-day)
BATCH_SIZE = 64
EPOCHS_BASE = 80        # for pooled pretraining
EPOCHS_FINETUNE = 30
LR = 1e-3
POOL_TRAIN_FRAC = 0.8   # fraction of pooled data used for training (walk-forward rather than shuffle)
MODEL_NAME_BASE = os.path.join(OUTPUT_DIR, "base_lstm.h5")

# Mode: 'pooled' to train base model on pooled assets, 'finetune' to fine-tune for a specific symbol,
# or 'both' to do pooled then finetune on chosen symbol.
MODE = "both"
FINETUNE_SYMBOL = "RELIANCE"   # change as required (symbol must be present in merged CSV)

# -----------------------
# UTIL: Attention layer (small)
# -----------------------
class AttentionLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], input_shape[-1]),
                                 initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(input_shape[-1],),
                                 initializer="zeros", trainable=True)
        self.u = self.add_weight(name="att_u", shape=(input_shape[-1],),
                                 initializer="glorot_uniform", trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        # inputs: (batch, time_steps, features)
        v = tf.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        vu = tf.tensordot(v, self.u, axes=1)  # (batch, time_steps)
        alphas = tf.nn.softmax(vu, axis=1)
        output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), axis=1)  # (batch, features)
        return output

# -----------------------
# FEATURES: technical indicators
# -----------------------
def add_technical_indicators(df):
    # Assumes df has columns: Date, Open, High, Low, Close, Adj Close, Volume
    df = df.copy()
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    # Basic returns
    df["return_1"] = df["Close"].pct_change().fillna(0)
    df["log_return_1"] = np.log1p(df["return_1"])

    # Moving averages
    df["ma7"] = df["Close"].rolling(7, min_periods=1).mean()
    df["ma21"] = df["Close"].rolling(21, min_periods=1).mean()
    df["ema12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["ema26"] = df["Close"].ewm(span=26, adjust=False).mean()

    # Volatility
    df["vol_7"] = df["return_1"].rolling(7, min_periods=1).std().fillna(0)
    df["vol_21"] = df["return_1"].rolling(21, min_periods=1).std().fillna(0)

    # MACD
    df["macd"] = df["ema12"] - df["ema26"]

    # RSI (approx implementation if ta not available)
    if ta is not None:
        df["rsi14"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
    else:
        delta = df["Close"].diff()
        up = delta.clip(lower=0).rolling(14, min_periods=1).mean()
        down = -delta.clip(upper=0).rolling(14, min_periods=1).mean()
        rs = up / (down + 1e-8)
        df["rsi14"] = 100 - (100 / (1 + rs))

    # Fill NaNs
    df.fillna(method="ffill", inplace=True)
    df.fillna(method="bfill", inplace=True)
    df.fillna(0, inplace=True)
    return df

# -----------------------
# Prepare sequences for one symbol
# -----------------------
def create_sequences(df, lookback=LOOKBACK, horizon=HORIZON, feature_cols=None):
    """
    df must be sorted by Date ascending and contain the features and target compute.
    target = future ROI% after `horizon` days: 100 * (Close[t+horizon] - Close[t]) / Close[t]
    """
    df = df.copy().reset_index(drop=True)
    # target ROI%
    df["target_roi"] = 100.0 * (df["Close"].shift(-horizon) - df["Close"]) / (df["Close"] + 1e-9)
    # drop last horizon rows with NaN targets
    df = df.iloc[:-horizon].reset_index(drop=True)

    if feature_cols is None:
        # pick features (excluding Date, Symbol, target)
        excluded = {"Date", "Symbol", "target_roi"}
        feature_cols = [c for c in df.columns if c not in excluded]

    X, y, idx_dates = [], [], []
    for i in range(len(df) - lookback + 1):
        seq = df.loc[i:i + lookback - 1, feature_cols].values
        target = df.loc[i + lookback - 1, "target_roi"]  # predict from last day of window
        X.append(seq)
        y.append(target)
        idx_dates.append(df.loc[i + lookback - 1, "Date"])
    X = np.array(X)
    y = np.array(y)
    return X, y, feature_cols, idx_dates

# -----------------------
# Model builder
# -----------------------
def build_model(input_shape, use_attention=True, hidden_units=(128, 64), lr=LR):
    K.clear_session()
    inp = layers.Input(shape=input_shape)
    x = layers.Masking(mask_value=0.0)(inp)
    x = layers.Bidirectional(layers.LSTM(hidden_units[0], return_sequences=True))(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Bidirectional(layers.LSTM(hidden_units[1], return_sequences=(use_attention)))(x)
    x = layers.Dropout(0.15)(x)
    if use_attention:
        x = AttentionLayer()(x)  # (batch, features)
    else:
        x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    out = layers.Dense(1, activation="linear")(x)
    model = models.Model(inputs=inp, outputs=out)
    optimizer = optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    return model

# -----------------------
# Train/evaluate helpers
# -----------------------
def train_model(X_train, y_train, X_val, y_val, model_save_path, epochs=EPOCHS_BASE):
    model = build_model(input_shape=X_train.shape[1:], use_attention=True)
    callbacks_list = [
        callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, verbose=1, min_lr=1e-6),
        callbacks.ModelCheckpoint(model_save_path, monitor="val_loss", save_best_only=True, verbose=1)
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        callbacks=callbacks_list,
        verbose=2,
        shuffle=False
    )
    return model, history

def evaluate_model(model, X, y):
    preds = model.predict(X).squeeze()
    mse = mean_squared_error(y, preds)
    mae = mean_absolute_error(y, preds)
    # MAPE with small epsilon
    mape = np.mean(np.abs((y - preds) / (np.abs(y) + 1e-9))) * 100.0
    # directional accuracy
    dir_acc = np.mean((np.sign(y) == np.sign(preds)).astype(float))
    return {"mse": mse, "mae": mae, "mape": mape, "dir_acc": dir_acc, "preds": preds}

# -----------------------
# Main pipeline
# -----------------------
def main():
    print(f"[{datetime.now()}] Loading data from {DATA_PATH} ...")
    df_all = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    df_all.sort_values(["Symbol", "Date"], inplace=True)

    # Filter symbols with enough history
    grouped = df_all.groupby("Symbol")
    min_required = LOOKBACK + HORIZON + 5
    symbols = [s for s, g in grouped if len(g) >= min_required]
    print(f"Symbols available with >= {min_required} rows: {len(symbols)}")

    # Build pooled dataset (concatenate sliding windows from all symbols)
    pooled_X_list, pooled_y_list = [], []
    pooled_feature_cols = None

    for sym in tqdm(symbols, desc="Preprocessing symbols"):
        g = grouped.get_group(sym).copy()
        g = add_technical_indicators(g)
        # normalize numeric columns before sequence generation? We'll fit scaler later per pooled set.
        X, y, feature_cols, dates = create_sequences(g, LOOKBACK, HORIZON)
        if pooled_feature_cols is None:
            pooled_feature_cols = feature_cols
        # append
        pooled_X_list.append(X)
        pooled_y_list.append(y)

    if not pooled_X_list:
        raise RuntimeError("No symbol has enough data. Increase data length or reduce LOOKBACK/HORIZON.")

    pooled_X = np.concatenate(pooled_X_list, axis=0)
    pooled_y = np.concatenate(pooled_y_list, axis=0)
    print(f"Pooled X shape: {pooled_X.shape}, pooled y shape: {pooled_y.shape}")

    # Flatten features for scaler (fit scaler on pooled train split)
    n_samples, timesteps, n_features = pooled_X.shape
    pooled_flat = pooled_X.reshape(-1, n_features)

    # Walk-forward train/val split by time (to simulate realistic forecasting)
    split_idx = int(n_samples * POOL_TRAIN_FRAC)
    train_flat = pooled_flat[: split_idx * timesteps]
    val_flat = pooled_flat[split_idx * timesteps :]

    scaler = StandardScaler().fit(train_flat)
    joblib.dump(scaler, os.path.join(SCALER_DIR, "scaler_base.pkl"))
    print("Saved base scaler.")

    # transform pooled dataset
    pooled_flat_scaled = scaler.transform(pooled_flat)
    pooled_X_scaled = pooled_flat_scaled.reshape(n_samples, timesteps, n_features)

    # split pooled
    X_train = pooled_X_scaled[:split_idx]
    y_train = pooled_y[:split_idx]
    X_val = pooled_X_scaled[split_idx:]
    y_val = pooled_y[split_idx:]

    print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")

    # --- MODE: pooled training ---
    if MODE in ("pooled", "both"):
        print("[INFO] Training base pooled model ...")
        model_base, hist = train_model(X_train, y_train, X_val, y_val, MODEL_NAME_BASE, epochs=EPOCHS_BASE)
        print("[INFO] Base training finished. Evaluating on val set ...")
        eval_val = evaluate_model(model_base, X_val, y_val)
        print(f"Base model Val metrics: MSE={eval_val['mse']:.6f}, MAE={eval_val['mae']:.6f}, MAPE={eval_val['mape']:.4f}%, DirAcc={eval_val['dir_acc']:.3f}")
    else:
        # if not training base, attempt to load it
        print("[INFO] Loading pretrained base model ...")
        model_base = tf.keras.models.load_model(MODEL_NAME_BASE, custom_objects={"AttentionLayer": AttentionLayer})

    # -----------------------
    # FINETUNE on single symbol
    # -----------------------
    if MODE in ("finetune", "both"):
        if FINETUNE_SYMBOL not in symbols:
            raise RuntimeError(f"Requested finetune symbol {FINETUNE_SYMBOL} not available in data (available: {symbols[:10]}...)")
        print(f"[INFO] Fine-tuning for {FINETUNE_SYMBOL} ...")
        g = grouped.get_group(FINETUNE_SYMBOL).copy()
        g = add_technical_indicators(g)
        X_sym, y_sym, feature_cols_sym, dates_sym = create_sequences(g, LOOKBACK, HORIZON)

        # scale using pooled scaler (recommended) or fit new scaler for symbol
        scaler_sym = joblib.load(os.path.join(SCALER_DIR, "scaler_base.pkl"))
        X_sym_flat = X_sym.reshape(-1, n_features)
        X_sym_scaled = scaler_sym.transform(X_sym_flat).reshape(X_sym.shape)

        # split by time
        split_sym = int(len(X_sym_scaled) * 0.8)
        X_sym_train, y_sym_train = X_sym_scaled[:split_sym], y_sym[:split_sym]
        X_sym_val, y_sym_val = X_sym_scaled[split_sym:], y_sym[split_sym:]

        # prepare model for finetuning: load base, freeze lower layers
        base = model_base
        # clone architecture to avoid accidental stateful issues
        finetune_model = build_model(input_shape=X_sym_train.shape[1:], use_attention=True)
        finetune_model.set_weights(base.get_weights())
        # freeze first LSTM layer group (by layer index) - allow top to adapt
        for layer in finetune_model.layers[:-4]:
            layer.trainable = False
        finetune_model.compile(optimizer=optimizers.Adam(learning_rate=LR/5), loss="mse", metrics=["mae"])

        finetune_path = os.path.join(OUTPUT_DIR, f"finetuned_{FINETUNE_SYMBOL}.h5")
        callbacks_list = [
            callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1, min_lr=1e-7),
            callbacks.ModelCheckpoint(finetune_path, monitor="val_loss", save_best_only=True, verbose=1)
        ]

        history = finetune_model.fit(
            X_sym_train, y_sym_train,
            validation_data=(X_sym_val, y_sym_val),
            epochs=EPOCHS_FINETUNE,
            batch_size=max(8, BATCH_SIZE//4),
            callbacks=callbacks_list,
            shuffle=False,
            verbose=2
        )

        # evaluate
        eval_train = evaluate_model(finetune_model, X_sym_train, y_sym_train)
        eval_val = evaluate_model(finetune_model, X_sym_val, y_sym_val)
        print(f"[FINETUNE] {FINETUNE_SYMBOL} Train MSE={eval_train['mse']:.6f} MAE={eval_train['mae']:.6f} MAPE={eval_train['mape']:.4f}% DirAcc={eval_train['dir_acc']:.3f}")
        print(f"[FINETUNE] {FINETUNE_SYMBOL} Val   MSE={eval_val['mse']:.6f} MAE={eval_val['mae']:.6f} MAPE={eval_val['mape']:.4f}% DirAcc={eval_val['dir_acc']:.3f}")

        # save scaler for symbol (same as base scaler)
        joblib.dump(scaler_sym, os.path.join(SCALER_DIR, f"scaler_{FINETUNE_SYMBOL}.pkl"))
        print(f"Saved finetuned model to {finetune_path} and scaler.")

    print("[DONE]")

if __name__ == "__main__":
    main()
