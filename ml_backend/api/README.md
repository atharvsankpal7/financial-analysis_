# 🚀 Financial Analysis Prediction API

A production-ready FastAPI application for predicting stock and commodity Return on Investment (ROI) using a hybrid LSTM-Attention deep learning model.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [API Endpoints](#-api-endpoints)
- [Usage Examples](#-usage-examples)
- [Model Details](#-model-details)
- [Configuration](#-configuration)
- [Deployment](#-deployment)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)

---

## ✨ Features

### 🎯 Core Capabilities
- **Multi-Asset Prediction**: Support for 20+ Nifty 50 stocks and 3 commodities (Gold, Silver, Platinum)
- **Advanced ML Model**: Hybrid Bidirectional LSTM with Multi-Head Attention mechanism
- **Real-time Data**: Fetches latest market data from Yahoo Finance
- **Batch Processing**: Predict multiple assets in a single API call
- **High Performance**: Optimized with TensorFlow for fast inference
- **Production Ready**: Comprehensive error handling, logging, and validation

### 🛡️ Reliability
- **Automatic Data Validation**: Pydantic models ensure data integrity
- **Graceful Error Handling**: Detailed error messages for debugging
- **Health Monitoring**: Built-in health check endpoint
- **CORS Support**: Configure for your frontend application

### 📊 API Features
- **Interactive Documentation**: Auto-generated Swagger UI and ReDoc
- **Type Safety**: Full type hints and validation
- **Versioned API**: Clear versioning for backward compatibility
- **Logging**: Comprehensive logging for monitoring and debugging

---

## 🏗️ Architecture

### System Overview

```
┌─────────────────┐
│   Client App    │
│  (Frontend/CLI) │
└────────┬────────┘
         │ HTTP/REST
         ▼
┌─────────────────────────────────────────┐
│         FastAPI Application             │
│  ┌───────────────────────────────────┐  │
│  │  Endpoints                        │  │
│  │  • /predict (Single)              │  │
│  │  • /predict/batch (Multiple)      │  │
│  │  • /health                        │  │
│  │  • /model/info                    │  │
│  └───────────────────────────────────┘  │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │  Data Processing Pipeline         │  │
│  │  1. Fetch from Yahoo Finance      │  │
│  │  2. Calculate ROI                 │  │
│  │  3. Log Transform                 │  │
│  │  4. Rolling Mean (5-day)          │  │
│  │  5. Min-Max Scaling               │  │
│  │  6. Sequence Creation (60 days)   │  │
│  └───────────────────────────────────┘  │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │  ML Model (TensorFlow/Keras)      │  │
│  │  • Bidirectional LSTM (128)       │  │
│  │  • Bidirectional LSTM (64)        │  │
│  │  • Multi-Head Attention (4 heads) │  │
│  │  • Dense Output Layer             │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│   Predictions   │
│  (JSON Response)│
└─────────────────┘
```

### Model Architecture

```
Input (60, n_assets)
    ↓
Bidirectional LSTM (128 units, dropout=0.2)
    ↓
Bidirectional LSTM (64 units, dropout=0.2)
    ↓
Multi-Head Attention (4 heads, key_dim=64)
    ↓
Add (Residual Connection) + Layer Normalization
    ↓
Global Average Pooling 1D
    ↓
Dense (128, ReLU, dropout=0.3)
    ↓
Dense (n_assets, Linear)
    ↓
Output (ROI Predictions)
```

---

## 📦 Installation

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Virtual environment (recommended)
- 4GB+ RAM recommended
- Trained model file (`hybrid_lstm_attention.keras`)

### Step 1: Clone or Navigate to Directory

```bash
cd ml_backend/api
```

### Step 2: Create Virtual Environment

**Windows:**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import fastapi, tensorflow; print('✅ Installation successful!')"
```

---

## 🚀 Quick Start

### 1. Start the API Server

```bash
python main.py
```

Or using uvicorn directly:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Verify Server is Running

Open your browser and navigate to:
- **API Documentation**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

### 3. Make Your First Prediction

Using curl:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "asset": "RELIANCE.NS",
    "days_ahead": 1
  }'
```

Using Python:
```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"asset": "RELIANCE.NS", "days_ahead": 1}
)
print(response.json())
```

---

## 🔌 API Endpoints

### 1. Root Endpoint
```http
GET /
```
Returns welcome message and API information.

**Response:**
```json
{
  "message": "Financial Analysis Prediction API",
  "version": "1.0.0",
  "docs": "/docs",
  "health": "/health"
}
```

---

### 2. Health Check
```http
GET /health
```
Check API health and model loading status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2025-11-02T10:30:00",
  "version": "1.0.0"
}
```

---

### 3. Model Information
```http
GET /model/info
```
Get detailed information about the ML model.

**Response:**
```json
{
  "model_type": "Hybrid LSTM-Attention",
  "architecture": "Bidirectional LSTM (128, 64) + Multi-Head Attention (4 heads)",
  "sequence_length": 60,
  "supported_assets": ["RELIANCE.NS", "TCS.NS", ...],
  "total_parameters": 245632,
  "last_trained": "2025-11"
}
```

---

### 4. Single Asset Prediction
```http
POST /predict
```

**Request Body:**
```json
{
  "asset": "RELIANCE.NS",
  "days_ahead": 1
}
```

**Parameters:**
- `asset` (string, required): Stock symbol (e.g., "RELIANCE.NS") or commodity ("GOLD")
- `days_ahead` (integer, optional): Days to predict ahead (1-30, default: 1)

**Response:**
```json
{
  "asset": "RELIANCE.NS",
  "current_roi": 1.25,
  "predicted_roi": 1.48,
  "predicted_change_percent": 18.4,
  "confidence": "High",
  "prediction_date": "2025-11-02T10:30:00",
  "days_ahead": 1
}
```

**Status Codes:**
- `200`: Successful prediction
- `400`: Invalid request (bad asset symbol, invalid days_ahead)
- `503`: Service unavailable (model not loaded, data fetch failed)
- `500`: Internal server error

---

### 5. Batch Prediction
```http
POST /predict/batch
```

**Request Body:**
```json
{
  "assets": ["RELIANCE.NS", "TCS.NS", "GOLD"],
  "days_ahead": 1
}
```

**Parameters:**
- `assets` (array, required): List of asset symbols (max 50)
- `days_ahead` (integer, optional): Days to predict ahead (1-30, default: 1)

**Response:**
```json
{
  "predictions": [
    {
      "asset": "RELIANCE.NS",
      "current_roi": 1.25,
      "predicted_roi": 1.48,
      "predicted_change_percent": 18.4,
      "confidence": "High",
      "prediction_date": "2025-11-02T10:30:00",
      "days_ahead": 1
    },
    {
      "asset": "TCS.NS",
      "current_roi": 0.87,
      "predicted_roi": 0.92,
      "predicted_change_percent": 5.7,
      "confidence": "High",
      "prediction_date": "2025-11-02T10:30:00",
      "days_ahead": 1
    }
  ],
  "total_assets": 2,
  "timestamp": "2025-11-02T10:30:00"
}
```

---

### 6. Supported Assets
```http
GET /assets/supported
```

**Response:**
```json
{
  "stocks": [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS",
    "ICICIBANK.NS", "LT.NS", "SBIN.NS", ...
  ],
  "commodities": ["GOLD", "SILVER", "PLATINUM"],
  "total": 23
}
```

---

## 💡 Usage Examples

### Python Examples

#### Example 1: Simple Prediction
```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "asset": "RELIANCE.NS",
        "days_ahead": 1
    }
)

if response.status_code == 200:
    data = response.json()
    print(f"Asset: {data['asset']}")
    print(f"Current ROI: {data['current_roi']:.2f}%")
    print(f"Predicted ROI: {data['predicted_roi']:.2f}%")
    print(f"Expected Change: {data['predicted_change_percent']:.2f}%")
    print(f"Confidence: {data['confidence']}")
else:
    print(f"Error: {response.status_code}")
    print(response.json())
```

#### Example 2: Batch Prediction
```python
import requests
import pandas as pd

# Predict multiple assets
assets = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "GOLD"]

response = requests.post(
    "http://localhost:8000/predict/batch",
    json={
        "assets": assets,
        "days_ahead": 1
    }
)

if response.status_code == 200:
    data = response.json()
    
    # Convert to DataFrame for easy analysis
    df = pd.DataFrame(data['predictions'])
    print(df[['asset', 'current_roi', 'predicted_roi', 'predicted_change_percent']])
    
    # Find best opportunities
    best_performers = df.nlargest(3, 'predicted_change_percent')
    print("\n🚀 Top 3 Predicted Performers:")
    print(best_performers[['asset', 'predicted_change_percent']])
```

#### Example 3: Error Handling
```python
import requests

def predict_with_retry(asset, max_retries=3):
    """Predict with automatic retry logic"""
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "http://localhost:8000/predict",
                json={"asset": asset, "days_ahead": 1},
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 503:
                print(f"Service unavailable, retrying... ({attempt+1}/{max_retries})")
                time.sleep(2)
            else:
                print(f"Error {response.status_code}: {response.json()}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"Network error: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
    
    return None

# Usage
result = predict_with_retry("RELIANCE.NS")
if result:
    print(f"Prediction successful: {result}")
```

### JavaScript/Node.js Example

```javascript
const axios = require('axios');

async function predictAsset(asset, daysAhead = 1) {
    try {
        const response = await axios.post('http://localhost:8000/predict', {
            asset: asset,
            days_ahead: daysAhead
        });
        
        const data = response.data;
        console.log(`Asset: ${data.asset}`);
        console.log(`Current ROI: ${data.current_roi.toFixed(2)}%`);
        console.log(`Predicted ROI: ${data.predicted_roi.toFixed(2)}%`);
        console.log(`Expected Change: ${data.predicted_change_percent.toFixed(2)}%`);
        console.log(`Confidence: ${data.confidence}`);
        
        return data;
    } catch (error) {
        if (error.response) {
            console.error(`Error ${error.response.status}:`, error.response.data);
        } else {
            console.error('Network error:', error.message);
        }
        return null;
    }
}

// Usage
predictAsset('RELIANCE.NS', 1);
```

### cURL Examples

```bash
# Single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"asset": "RELIANCE.NS", "days_ahead": 1}'

# Batch prediction
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "assets": ["RELIANCE.NS", "TCS.NS", "GOLD"],
    "days_ahead": 1
  }'

# Get supported assets
curl "http://localhost:8000/assets/supported"

# Health check
curl "http://localhost:8000/health"

# Model info
curl "http://localhost:8000/model/info"
```

---

## 🧠 Model Details

### Training Data
- **Source**: Yahoo Finance
- **Period**: 2.5 years of historical data
- **Assets**: 20+ Nifty 50 stocks + 3 commodities
- **Features**: Return on Investment (ROI) percentages
- **Frequency**: Daily data

### Preprocessing Pipeline

1. **Data Collection**: Fetch Adj Close prices from Yahoo Finance
2. **ROI Calculation**: `ROI = (Price_t - Price_t-1) / Price_t-1 * 100`
3. **Log Transformation**: `log1p(ROI)` for numerical stability
4. **Smoothing**: 5-day rolling mean to reduce noise
5. **Normalization**: Min-Max scaling to [0, 1] range
6. **Sequencing**: Create 60-day sliding windows

### Model Specifications

| Component | Details |
|-----------|---------|
| **Input Shape** | (60, n_assets) |
| **Layer 1** | Bidirectional LSTM (128 units, dropout=0.2) |
| **Layer 2** | Bidirectional LSTM (64 units, dropout=0.2) |
| **Attention** | Multi-Head Attention (4 heads, key_dim=64) |
| **Normalization** | Layer Normalization |
| **Pooling** | Global Average Pooling 1D |
| **Dense 1** | 128 units, ReLU, dropout=0.3 |
| **Output** | n_assets units, Linear activation |
| **Optimizer** | AdamW (lr=1e-3, weight_decay=1e-4) |
| **Loss** | Mean Squared Error (MSE) |
| **Metrics** | MAE, RMSE, R² |

### Performance Metrics

Typical performance on test data:
- **MAE (Mean Absolute Error)**: ~0.02-0.05
- **RMSE (Root Mean Squared Error)**: ~0.03-0.07
- **R² Score**: ~0.65-0.85

*Note: Performance varies by asset and market conditions*

### Confidence Levels

The API provides confidence levels based on prediction magnitude:

| Change | Confidence Level |
|--------|-----------------|
| < 0.5% | High |
| 0.5% - 2.0% | Medium |
| > 2.0% | Low |

---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the `api` directory:

```env
# Server Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=true

# Model Configuration
MODEL_PATH=../train3/models/hybrid_lstm_attention.keras
DATASET_PATH=../train3/dataset_combined_2_5yr.csv

# Sequence Configuration
SEQ_LEN=60
ROLLING_WINDOW=5

# CORS Configuration
CORS_ORIGINS=["http://localhost:3000", "http://localhost:5173"]

# Logging
LOG_LEVEL=INFO
```

### Modifying Configuration

Edit `main.py` to customize:

```python
class Config:
    MODEL_PATH = "path/to/your/model.keras"
    DATASET_PATH = "path/to/your/dataset.csv"
    SEQ_LEN = 60  # Sequence length for predictions
    ROLLING_WINDOW = 5  # Smoothing window
    
    # Add more stocks
    SUPPORTED_STOCKS = [
        "RELIANCE.NS",
        "TCS.NS",
        # ... add more
    ]
```

---

## 🚢 Deployment

### Local Development

```bash
# Development mode with auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Production Deployment

#### Option 1: Using Gunicorn (Recommended for Production)

```bash
# Install gunicorn
pip install gunicorn

# Run with 4 workers
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

#### Option 2: Using Docker

Create `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:

```bash
docker build -t financial-api .
docker run -p 8000:8000 financial-api
```

#### Option 3: Using Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
    environment:
      - LOG_LEVEL=INFO
    restart: unless-stopped
```

Run:

```bash
docker-compose up -d
```

### Cloud Deployment

#### Deploy to AWS EC2

1. Launch an EC2 instance (t2.medium or larger recommended)
2. Install Python and dependencies
3. Clone your repository
4. Set up systemd service:

```ini
# /etc/systemd/system/financial-api.service
[Unit]
Description=Financial Analysis API
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/financial-analysis_/ml_backend/api
ExecStart=/home/ubuntu/financial-analysis_/ml_backend/api/venv/bin/gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
Restart=always

[Install]
WantedBy=multi-user.target
```

5. Start the service:

```bash
sudo systemctl enable financial-api
sudo systemctl start financial-api
```

#### Deploy to Heroku

Create `Procfile`:

```
web: gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT
```

Deploy:

```bash
heroku create your-app-name
git push heroku main
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. Model Not Loading

**Error**: `Model file not found`

**Solution**:
```bash
# Check if model exists
ls ../train3/models/hybrid_lstm_attention.keras

# Update MODEL_PATH in Config if needed
```

#### 2. TensorFlow Import Error

**Error**: `ModuleNotFoundError: No module named 'tensorflow'`

**Solution**:
```bash
# Reinstall TensorFlow
pip install --upgrade tensorflow==2.15.0
```

#### 3. Yahoo Finance Data Fetch Fails

**Error**: `Failed to fetch market data`

**Solution**:
- Check internet connection
- Verify asset symbol is correct
- Yahoo Finance may be temporarily unavailable - retry after a few minutes

#### 4. Memory Issues

**Error**: `MemoryError` or `Out of memory`

**Solution**:
```python
# Reduce batch size or enable memory growth
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```

#### 5. Port Already in Use

**Error**: `Address already in use`

**Solution**:
```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/macOS
lsof -ti:8000 | xargs kill -9
```

### Debug Mode

Enable detailed logging:

```python
# In main.py
logging.basicConfig(level=logging.DEBUG)
```

### Testing Endpoints

Use the interactive documentation at `http://localhost:8000/docs` to test endpoints directly in your browser.

---

## 📚 Additional Resources

### API Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

### Related Documentation
- [Training Guide](../train3/TRAINING_README.md) - Model training documentation
- [Data Collection Guide](../data_scapping/README.md) - Data scraping documentation
- [Project Overview](../../README.md) - Main project README

### External Resources
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [TensorFlow Guide](https://www.tensorflow.org/guide)
- [Yahoo Finance API](https://python-yahoofinance.readthedocs.io/)

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👥 Support

For issues, questions, or suggestions:

- **Email**: support@financial-analysis.com
- **Issues**: [GitHub Issues](https://github.com/yourusername/financial-analysis_/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/financial-analysis_/discussions)

---

## 🙏 Acknowledgments

- Yahoo Finance for providing market data
- TensorFlow team for the ML framework
- FastAPI team for the excellent web framework
- Nifty 50 for stock listings

---

**Made with ❤️ by Financial Analysis Team**

*Last Updated: November 2025*
