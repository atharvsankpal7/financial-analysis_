# 🧠 Model Training Guide - Hybrid LSTM-Attention

Comprehensive documentation for training the Financial Analysis prediction model using deep learning.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)
![LSTM](https://img.shields.io/badge/Model-LSTM--Attention-green.svg)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Model Architecture](#-model-architecture)
- [Data Preparation](#-data-preparation)
- [Training Process](#-training-process)
- [Quick Start](#-quick-start)
- [Hyperparameters](#-hyperparameters)
- [Evaluation Metrics](#-evaluation-metrics)
- [Optimization Strategies](#-optimization-strategies)
- [Troubleshooting](#-troubleshooting)
- [Advanced Configuration](#-advanced-configuration)

---

## 🎯 Overview

### What is This Model?

This is a **Hybrid LSTM-Attention** neural network designed to predict Return on Investment (ROI) for multiple financial assets simultaneously. The model combines:

1. **Bidirectional LSTM**: Captures temporal patterns in both forward and backward directions
2. **Multi-Head Attention**: Identifies important time steps and relationships
3. **Residual Connections**: Improves gradient flow and training stability
4. **Layer Normalization**: Stabilizes training and speeds convergence

### Key Features

- ✅ **Multi-Asset Prediction**: Predicts 20+ stocks and commodities simultaneously
- ✅ **Time-Series Forecasting**: Uses 60-day sequences for predictions
- ✅ **Robust Architecture**: Dropout, normalization, and regularization
- ✅ **Production-Ready**: Saves models in Keras format for easy deployment
- ✅ **Comprehensive Monitoring**: TensorBoard integration and detailed metrics

### Performance Expectations

| Metric | Target | Typical |
|--------|--------|---------|
| **MAE** | < 0.05 | 0.02-0.04 |
| **RMSE** | < 0.10 | 0.03-0.07 |
| **R² Score** | > 0.70 | 0.65-0.85 |
| **Training Time** | - | 20-40 minutes |

---

## 🏗️ Model Architecture

### Network Diagram

```
┌─────────────────────────────────────────────────────┐
│                  INPUT LAYER                        │
│              Shape: (60, n_assets)                  │
│         (60 time steps × n_assets features)         │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│         BIDIRECTIONAL LSTM LAYER 1                  │
│              Units: 128 (64 + 64)                   │
│              Dropout: 0.2                           │
│              Return Sequences: True                 │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│         BIDIRECTIONAL LSTM LAYER 2                  │
│              Units: 64 (32 + 32)                    │
│              Dropout: 0.2                           │
│              Return Sequences: True                 │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│         MULTI-HEAD ATTENTION LAYER                  │
│              Heads: 4                               │
│              Key Dimension: 64                      │
│              Query = Key = Value                    │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│         RESIDUAL CONNECTION                         │
│         Add(LSTM_output, Attention_output)          │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│         LAYER NORMALIZATION                         │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│         GLOBAL AVERAGE POOLING 1D                   │
│         (Reduces sequence dimension)                │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│         DENSE LAYER                                 │
│              Units: 128                             │
│              Activation: ReLU                       │
│              Dropout: 0.3                           │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│         OUTPUT LAYER                                │
│              Units: n_assets                        │
│              Activation: Linear                     │
│         (Predicts ROI for each asset)               │
└─────────────────────────────────────────────────────┘
```

### Layer-by-Layer Explanation

#### 1. Input Layer
- **Shape**: `(batch_size, 60, n_assets)`
- **Purpose**: Accept 60-day sequences of normalized ROI values
- **Features**: Each asset's ROI percentage

#### 2. Bidirectional LSTM Layer 1
- **Forward LSTM**: 64 units (processes sequence left-to-right)
- **Backward LSTM**: 64 units (processes sequence right-to-left)
- **Combined Output**: 128 units
- **Dropout**: 20% to prevent overfitting
- **Purpose**: Capture long-term temporal dependencies

#### 3. Bidirectional LSTM Layer 2
- **Forward LSTM**: 32 units
- **Backward LSTM**: 32 units
- **Combined Output**: 64 units
- **Purpose**: Refine temporal features at a higher abstraction level

#### 4. Multi-Head Attention
- **Mechanism**: Self-attention over the sequence
- **Heads**: 4 parallel attention mechanisms
- **Key Dimension**: 64
- **Purpose**: Learn which time steps are most important for prediction

#### 5. Residual Connection & Normalization
- **Add Layer**: Combines LSTM output with attention output
- **Layer Norm**: Normalizes the combined output
- **Purpose**: Improve gradient flow and stabilize training

#### 6. Global Average Pooling
- **Operation**: Average across time dimension
- **Output**: Fixed-size vector regardless of sequence length
- **Purpose**: Aggregate temporal information

#### 7. Dense Layer
- **Units**: 128
- **Activation**: ReLU
- **Dropout**: 30%
- **Purpose**: Learn complex non-linear relationships

#### 8. Output Layer
- **Units**: n_assets (number of assets being predicted)
- **Activation**: Linear (for regression)
- **Purpose**: Produce final ROI predictions

---

## 📊 Data Preparation

### Data Collection (`scap.py`)

The data collection script fetches historical data from Yahoo Finance:

```python
# Supported Assets
STOCKS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", 
    "ICICIBANK.NS", "LT.NS", "SBIN.NS", ...
]

COMMODITIES = {
    "Gold": "GC=F",
    "Silver": "SI=F", 
    "Platinum": "PL=F"
}

# Time Period
START_DATE = 2.5 years ago
END_DATE = Today
```

### Data Processing Pipeline

#### Step 1: Fetch Raw Data
```python
import yfinance as yf

# Fetch adjusted close prices
df = yf.download(symbol, start=start_date, end=end_date)
prices = df['Adj Close']
```

#### Step 2: Calculate ROI
```python
# ROI = (Price_t - Price_t-1) / Price_t-1 * 100
roi = prices.pct_change() * 100
```

#### Step 3: Handle Outliers
```python
# Replace infinite values with NaN
data = data.replace([np.inf, -np.inf], np.nan)

# Forward fill then backward fill missing values
data = data.ffill().bfill()
```

#### Step 4: Log Transformation
```python
# Apply log1p for numerical stability
# log1p(x) = log(1 + x), handles values near zero
data = np.log1p(data)
```

#### Step 5: Smoothing
```python
# 5-day rolling mean to reduce noise
data = data.rolling(5).mean().dropna()
```

#### Step 6: Normalization
```python
from sklearn.preprocessing import MinMaxScaler

# Scale to [0, 1] range
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(data)
```

#### Step 7: Sequence Creation
```python
def create_sequences(data, seq_len=60):
    """Create sliding window sequences"""
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i-seq_len:i])  # Previous 60 days
        y.append(data[i])             # Current day (target)
    return np.array(X), np.array(y)

X, y = create_sequences(scaled, seq_len=60)
```

### Dataset Structure

**Input (X)**:
- Shape: `(n_samples, 60, n_assets)`
- Example: `(1000, 60, 23)` for 1000 samples, 60 days, 23 assets

**Output (y)**:
- Shape: `(n_samples, n_assets)`
- Example: `(1000, 23)` - one ROI value per asset

**Train/Test Split**:
- Training: 80% of data (chronologically first)
- Testing: 20% of data (chronologically last)
- **Important**: No shuffling to maintain temporal order!

---

## 🚀 Training Process

### Quick Start Training

#### 1. Setup Environment

```bash
cd ml_backend/train3

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install tensorflow scikit-learn pandas numpy yfinance
```

#### 2. Collect Data

```bash
# Run data collection script
python scap.py
```

This creates `dataset_combined_2_5yr.csv` with all asset ROI data.

#### 3. Train Model

```bash
# Run training script
python train.py
```

The training process will:
1. Load and preprocess data
2. Create sequences
3. Build model
4. Train for up to 120 epochs
5. Save best model to `models/hybrid_lstm_attention.keras`

#### 4. Monitor Training

Watch the console output:

```
Epoch 1/120
25/25 [==============================] - 15s 450ms/step - loss: 0.0234 - mae: 0.1156 - val_loss: 0.0189 - val_mae: 0.0987
Epoch 2/120
25/25 [==============================] - 12s 420ms/step - loss: 0.0178 - mae: 0.0923 - val_loss: 0.0145 - val_mae: 0.0832
...
```

---

## ⚙️ Hyperparameters

### Model Hyperparameters

```python
# Architecture
SEQ_LEN = 60              # Sequence length (days)
LSTM_UNITS_1 = 128        # First LSTM layer units
LSTM_UNITS_2 = 64         # Second LSTM layer units
ATTENTION_HEADS = 4       # Multi-head attention heads
ATTENTION_KEY_DIM = 64    # Attention key dimension
DENSE_UNITS = 128         # Dense layer units
DROPOUT_LSTM = 0.2        # LSTM dropout rate
DROPOUT_DENSE = 0.3       # Dense dropout rate

# Training
EPOCHS = 120              # Maximum epochs
BATCH_SIZE = 32           # Samples per batch
LEARNING_RATE = 1e-3      # Initial learning rate
WEIGHT_DECAY = 1e-4       # L2 regularization
VALIDATION_SPLIT = 0.2    # 20% for validation
```

### Optimizer Configuration

```python
from tensorflow.keras.optimizers import AdamW

optimizer = AdamW(
    learning_rate=1e-3,    # Initial learning rate
    weight_decay=1e-4,     # L2 regularization
    beta_1=0.9,            # Exponential decay rate for 1st moment
    beta_2=0.999,          # Exponential decay rate for 2nd moment
    epsilon=1e-7           # Small constant for numerical stability
)
```

### Callbacks

```python
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint
)

callbacks = [
    # Reduce learning rate when validation loss plateaus
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,          # Reduce LR by 50%
        patience=5,          # After 5 epochs without improvement
        min_lr=1e-6,         # Minimum learning rate
        verbose=1
    ),
    
    # Stop training if no improvement
    EarlyStopping(
        monitor='val_loss',
        patience=10,         # Stop after 10 epochs without improvement
        restore_best_weights=True,
        verbose=1
    ),
    
    # Save best model
    ModelCheckpoint(
        'models/hybrid_lstm_attention.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
]
```

---

## 📈 Evaluation Metrics

### Metrics Explained

#### 1. Mean Absolute Error (MAE)
```python
MAE = (1/n) * Σ|predicted - actual|
```
- **What it means**: Average absolute difference between predictions and actual values
- **Good value**: < 0.05 (5% average error)
- **Units**: Same as target (ROI percentage)

#### 2. Root Mean Squared Error (RMSE)
```python
RMSE = √[(1/n) * Σ(predicted - actual)²]
```
- **What it means**: Square root of average squared errors (penalizes large errors more)
- **Good value**: < 0.10
- **Units**: Same as target (ROI percentage)

#### 3. R² Score (Coefficient of Determination)
```python
R² = 1 - (SS_residual / SS_total)
```
- **What it means**: Proportion of variance explained by the model
- **Range**: -∞ to 1 (1 is perfect prediction)
- **Good value**: > 0.70 (70% variance explained)

### Evaluation Code

```python
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score
)

# Make predictions
predictions = model.predict(X_test)

# Calculate metrics
mae = mean_absolute_error(y_test, predictions)
rmse = root_mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.3f}")
```

### Per-Asset Evaluation

```python
# Evaluate each asset separately
for i, asset in enumerate(asset_columns):
    mae_asset = mean_absolute_error(y_test[:, i], predictions[:, i])
    print(f"{asset}: MAE = {mae_asset:.4f}")
```

---

## 🔧 Optimization Strategies

### 1. Hyperparameter Tuning

#### Learning Rate Tuning
```python
# Try different learning rates
learning_rates = [1e-2, 1e-3, 1e-4, 1e-5]

for lr in learning_rates:
    model = build_hybrid_model(...)
    model.compile(optimizer=AdamW(learning_rate=lr), ...)
    history = model.fit(...)
    # Evaluate and compare
```

#### Batch Size Tuning
```python
# Smaller batch = more noise, better generalization
# Larger batch = faster training, more stable

batch_sizes = [16, 32, 64, 128]
```

#### Architecture Search
```python
# Try different LSTM units
lstm_configs = [
    (64, 32),   # Smaller model
    (128, 64),  # Current model
    (256, 128), # Larger model
]
```

### 2. Regularization Techniques

#### Dropout
```python
# Prevent overfitting by randomly dropping neurons
DROPOUT_LSTM = 0.2   # 20% dropout in LSTM
DROPOUT_DENSE = 0.3  # 30% dropout in Dense layer
```

#### Weight Decay (L2 Regularization)
```python
# Penalize large weights
optimizer = AdamW(weight_decay=1e-4)
```

#### Early Stopping
```python
# Stop when validation loss stops improving
EarlyStopping(patience=10, restore_best_weights=True)
```

### 3. Data Augmentation

#### Add Noise
```python
# Add small random noise to training data
noise = np.random.normal(0, 0.01, X_train.shape)
X_train_augmented = X_train + noise
```

#### Time Shifting
```python
# Create shifted sequences for more training samples
def augment_sequences(X, y, shifts=[-1, 0, 1]):
    X_aug, y_aug = [], []
    for shift in shifts:
        if shift == 0:
            X_aug.append(X)
            y_aug.append(y)
        # Add shifting logic
    return np.concatenate(X_aug), np.concatenate(y_aug)
```

### 4. Ensemble Methods

#### Model Averaging
```python
# Train multiple models and average predictions
models = []
for i in range(5):
    model = build_hybrid_model(...)
    model.fit(...)
    models.append(model)

# Predict
predictions = np.mean([m.predict(X_test) for m in models], axis=0)
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. Model Overfitting

**Symptoms**:
- Training loss decreases, validation loss increases
- Large gap between train and validation metrics

**Solutions**:
```python
# Increase dropout
DROPOUT_LSTM = 0.3
DROPOUT_DENSE = 0.4

# Add L2 regularization
from tensorflow.keras import regularizers
layers.Dense(128, kernel_regularizer=regularizers.l2(0.01))

# Use early stopping
EarlyStopping(patience=5)

# Get more data
# Reduce model complexity
```

#### 2. Model Underfitting

**Symptoms**:
- Both training and validation loss are high
- Model doesn't learn

**Solutions**:
```python
# Increase model capacity
LSTM_UNITS_1 = 256
LSTM_UNITS_2 = 128

# Train longer
EPOCHS = 200

# Reduce dropout
DROPOUT_LSTM = 0.1

# Check data quality
# Adjust learning rate
```

#### 3. Exploding/Vanishing Gradients

**Symptoms**:
- Loss becomes NaN or infinity
- Loss doesn't change

**Solutions**:
```python
# Use gradient clipping
optimizer = AdamW(learning_rate=1e-3, clipnorm=1.0)

# Reduce learning rate
LEARNING_RATE = 1e-4

# Use BatchNormalization
layers.BatchNormalization()

# Check for outliers in data
```

#### 4. Slow Training

**Solutions**:
```python
# Use GPU if available
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Increase batch size
BATCH_SIZE = 64

# Reduce model size
# Use mixed precision training
tf.keras.mixed_precision.set_global_policy('mixed_float16')
```

#### 5. Out of Memory

**Solutions**:
```python
# Reduce batch size
BATCH_SIZE = 16

# Reduce sequence length
SEQ_LEN = 30

# Enable memory growth for GPU
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```

---

## 🔬 Advanced Configuration

### Custom Loss Functions

```python
def weighted_mse(y_true, y_pred):
    """Give more weight to recent predictions"""
    weights = tf.range(1.0, tf.cast(tf.shape(y_true)[1], tf.float32) + 1)
    weights = weights / tf.reduce_sum(weights)
    
    mse = tf.square(y_true - y_pred)
    weighted = mse * weights
    return tf.reduce_mean(weighted)

model.compile(optimizer=opt, loss=weighted_mse)
```

### Transfer Learning

```python
# Load pre-trained model
base_model = tf.keras.models.load_model('models/hybrid_lstm_attention.keras')

# Freeze early layers
for layer in base_model.layers[:-3]:
    layer.trainable = False

# Fine-tune on new data
base_model.compile(...)
base_model.fit(X_new, y_new, epochs=20)
```

### Multi-Step Prediction

```python
def predict_multi_step(model, initial_sequence, steps=5):
    """Predict multiple steps ahead"""
    predictions = []
    current_sequence = initial_sequence.copy()
    
    for _ in range(steps):
        # Predict next step
        pred = model.predict(current_sequence.reshape(1, SEQ_LEN, -1))
        predictions.append(pred)
        
        # Update sequence (roll and append prediction)
        current_sequence = np.roll(current_sequence, -1, axis=0)
        current_sequence[-1] = pred
    
    return np.array(predictions)
```

### TensorBoard Integration

```python
from tensorflow.keras.callbacks import TensorBoard
import datetime

# Create log directory
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Add TensorBoard callback
tensorboard_callback = TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
    write_graph=True,
    write_images=True,
    update_freq='epoch'
)

# Train with TensorBoard
model.fit(..., callbacks=[tensorboard_callback, ...])

# View in browser
# tensorboard --logdir=logs/fit
```

---

## 📚 References

### Research Papers

1. **LSTM Networks**: Hochreiter & Schmidhuber (1997) - "Long Short-Term Memory"
2. **Attention Mechanism**: Vaswani et al. (2017) - "Attention Is All You Need"
3. **Bidirectional RNNs**: Schuster & Paliwal (1997) - "Bidirectional Recurrent Neural Networks"

### Libraries Documentation

- [TensorFlow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)
- [Scikit-learn](https://scikit-learn.org/)
- [Yahoo Finance](https://python-yahoofinance.readthedocs.io/)

---

## 📝 Training Checklist

Before training:
- [ ] Data collected (`dataset_combined_2_5yr.csv` exists)
- [ ] Virtual environment activated
- [ ] Dependencies installed
- [ ] GPU configured (if available)
- [ ] Sufficient disk space (>1GB)

During training:
- [ ] Monitor loss curves
- [ ] Check for overfitting
- [ ] Validate on test set
- [ ] Save best model

After training:
- [ ] Evaluate on test set
- [ ] Test API predictions
- [ ] Document results
- [ ] Version control model

---

## 🎓 Best Practices

1. **Always use validation set** to monitor overfitting
2. **Don't shuffle time-series data** - maintain temporal order
3. **Normalize features** to [0, 1] or standardize to mean=0, std=1
4. **Use early stopping** to prevent wasting compute
5. **Save best model** not just the final model
6. **Version your models** with timestamps or metrics
7. **Document hyperparameters** used for each training run
8. **Test on unseen data** before deployment

---

**Happy Training! 🚀**

*Last Updated: November 2025*
