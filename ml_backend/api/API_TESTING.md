# 📮 API Testing Guide - Postman Collection

Complete guide for testing the Financial Analysis Prediction API using Postman, cURL, or any HTTP client.

---

## 📋 Table of Contents

- [Postman Collection](#-postman-collection)
- [cURL Examples](#-curl-examples)
- [Python Examples](#-python-examples)
- [JavaScript Examples](#-javascript-examples)
- [Response Schemas](#-response-schemas)
- [Error Codes](#-error-codes)

---

## 🎯 Postman Collection

### Setup

1. **Create New Collection**
   - Name: "Financial Analysis API"
   - Base URL Variable: `{{base_url}}` = `http://localhost:8000`

2. **Add Requests** (detailed below)

---

### 1. Health Check

**Request**
```
GET {{base_url}}/health
```

**Headers**
```
(none required)
```

**Response** (200 OK)
```json
{
    "status": "healthy",
    "model_loaded": true,
    "timestamp": "2025-11-02T10:30:00.123456",
    "version": "1.0.0"
}
```

---

### 2. Root/Welcome

**Request**
```
GET {{base_url}}/
```

**Response** (200 OK)
```json
{
    "message": "Financial Analysis Prediction API",
    "version": "1.0.0",
    "docs": "/docs",
    "health": "/health"
}
```

---

### 3. Model Information

**Request**
```
GET {{base_url}}/model/info
```

**Response** (200 OK)
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

### 4. Single Prediction - Stock

**Request**
```
POST {{base_url}}/predict
```

**Headers**
```
Content-Type: application/json
```

**Body (raw JSON)**
```json
{
    "asset": "RELIANCE.NS",
    "days_ahead": 1
}
```

**Response** (200 OK)
```json
{
    "asset": "RELIANCE.NS",
    "current_roi": 1.25,
    "predicted_roi": 1.48,
    "predicted_change_percent": 18.4,
    "confidence": "High",
    "prediction_date": "2025-11-02T10:30:00.123456",
    "days_ahead": 1
}
```

---

### 5. Single Prediction - Commodity

**Request**
```
POST {{base_url}}/predict
```

**Headers**
```
Content-Type: application/json
```

**Body (raw JSON)**
```json
{
    "asset": "GOLD",
    "days_ahead": 1
}
```

**Response** (200 OK)
```json
{
    "asset": "GOLD",
    "current_roi": 0.87,
    "predicted_roi": 0.95,
    "predicted_change_percent": 9.2,
    "confidence": "High",
    "prediction_date": "2025-11-02T10:30:00.123456",
    "days_ahead": 1
}
```

---

### 6. Batch Prediction

**Request**
```
POST {{base_url}}/predict/batch
```

**Headers**
```
Content-Type: application/json
```

**Body (raw JSON)**
```json
{
    "assets": [
        "RELIANCE.NS",
        "TCS.NS",
        "INFY.NS",
        "GOLD",
        "SILVER"
    ],
    "days_ahead": 1
}
```

**Response** (200 OK)
```json
{
    "predictions": [
        {
            "asset": "RELIANCE.NS",
            "current_roi": 1.25,
            "predicted_roi": 1.48,
            "predicted_change_percent": 18.4,
            "confidence": "High",
            "prediction_date": "2025-11-02T10:30:00.123456",
            "days_ahead": 1
        },
        {
            "asset": "TCS.NS",
            "current_roi": 0.87,
            "predicted_roi": 0.92,
            "predicted_change_percent": 5.7,
            "confidence": "High",
            "prediction_date": "2025-11-02T10:30:00.123456",
            "days_ahead": 1
        }
    ],
    "total_assets": 5,
    "timestamp": "2025-11-02T10:30:00.123456"
}
```

---

### 7. Get Supported Assets

**Request**
```
GET {{base_url}}/assets/supported
```

**Response** (200 OK)
```json
{
    "stocks": [
        "RELIANCE.NS",
        "TCS.NS",
        "INFY.NS",
        "HDFCBANK.NS",
        "ICICIBANK.NS",
        "LT.NS",
        "SBIN.NS",
        "BHARTIARTL.NS",
        "HINDUNILVR.NS",
        "KOTAKBANK.NS",
        "ITC.NS",
        "BAJFINANCE.NS",
        "ASIANPAINT.NS",
        "MARUTI.NS",
        "WIPRO.NS",
        "AXISBANK.NS",
        "SUNPHARMA.NS",
        "TITAN.NS",
        "ONGC.NS",
        "ULTRACEMCO.NS"
    ],
    "commodities": [
        "GOLD",
        "SILVER",
        "PLATINUM"
    ],
    "total": 23
}
```

---

## 💻 cURL Examples

### Windows PowerShell

#### Health Check
```powershell
curl http://localhost:8000/health
```

#### Single Prediction
```powershell
curl -X POST "http://localhost:8000/predict" `
  -H "Content-Type: application/json" `
  -d '{"asset": "RELIANCE.NS", "days_ahead": 1}'
```

#### Batch Prediction
```powershell
curl -X POST "http://localhost:8000/predict/batch" `
  -H "Content-Type: application/json" `
  -d '{
    "assets": ["RELIANCE.NS", "TCS.NS", "GOLD"],
    "days_ahead": 1
  }'
```

### Linux/macOS

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Single Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"asset": "RELIANCE.NS", "days_ahead": 1}'
```

#### Batch Prediction
```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "assets": ["RELIANCE.NS", "TCS.NS", "GOLD"],
    "days_ahead": 1
  }'
```

---

## 🐍 Python Examples

### Using requests Library

```python
import requests
import json

# Base URL
BASE_URL = "http://localhost:8000"

# 1. Health Check
response = requests.get(f"{BASE_URL}/health")
print(response.json())

# 2. Single Prediction
payload = {
    "asset": "RELIANCE.NS",
    "days_ahead": 1
}
response = requests.post(f"{BASE_URL}/predict", json=payload)
print(response.json())

# 3. Batch Prediction
payload = {
    "assets": ["RELIANCE.NS", "TCS.NS", "INFY.NS"],
    "days_ahead": 1
}
response = requests.post(f"{BASE_URL}/predict/batch", json=payload)
print(json.dumps(response.json(), indent=2))

# 4. Get Supported Assets
response = requests.get(f"{BASE_URL}/assets/supported")
assets = response.json()
print(f"Total stocks: {len(assets['stocks'])}")
print(f"Total commodities: {len(assets['commodities'])}")
```

### With Error Handling

```python
import requests

def make_prediction(asset, days_ahead=1):
    """Make a prediction with error handling"""
    try:
        response = requests.post(
            "http://localhost:8000/predict",
            json={"asset": asset, "days_ahead": days_ahead},
            timeout=10
        )
        response.raise_for_status()  # Raise exception for bad status codes
        
        data = response.json()
        return {
            "success": True,
            "data": data
        }
        
    except requests.exceptions.HTTPError as e:
        return {
            "success": False,
            "error": f"HTTP Error: {e.response.status_code}",
            "detail": e.response.json() if e.response else None
        }
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "error": "Connection Error: Could not connect to API"
        }
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": "Timeout: Request took too long"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Unexpected Error: {str(e)}"
        }

# Usage
result = make_prediction("RELIANCE.NS")
if result["success"]:
    print(f"Prediction: {result['data']['predicted_roi']:.2f}%")
else:
    print(f"Error: {result['error']}")
```

---

## 🌐 JavaScript Examples

### Using Fetch API

```javascript
// Base URL
const BASE_URL = 'http://localhost:8000';

// 1. Health Check
async function checkHealth() {
    const response = await fetch(`${BASE_URL}/health`);
    const data = await response.json();
    console.log(data);
}

// 2. Single Prediction
async function predictSingle(asset, daysAhead = 1) {
    const response = await fetch(`${BASE_URL}/predict`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            asset: asset,
            days_ahead: daysAhead
        })
    });
    
    const data = await response.json();
    return data;
}

// 3. Batch Prediction
async function predictBatch(assets, daysAhead = 1) {
    const response = await fetch(`${BASE_URL}/predict/batch`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            assets: assets,
            days_ahead: daysAhead
        })
    });
    
    const data = await response.json();
    return data;
}

// Usage
checkHealth();

predictSingle('RELIANCE.NS')
    .then(data => console.log('Single prediction:', data))
    .catch(error => console.error('Error:', error));

predictBatch(['RELIANCE.NS', 'TCS.NS', 'GOLD'])
    .then(data => console.log('Batch predictions:', data))
    .catch(error => console.error('Error:', error));
```

### Using Axios

```javascript
const axios = require('axios');

const BASE_URL = 'http://localhost:8000';

// Single Prediction
async function predictAsset(asset) {
    try {
        const response = await axios.post(`${BASE_URL}/predict`, {
            asset: asset,
            days_ahead: 1
        });
        
        console.log(`Asset: ${response.data.asset}`);
        console.log(`Current ROI: ${response.data.current_roi}%`);
        console.log(`Predicted ROI: ${response.data.predicted_roi}%`);
        console.log(`Change: ${response.data.predicted_change_percent}%`);
        
        return response.data;
    } catch (error) {
        if (error.response) {
            console.error(`Error ${error.response.status}:`, error.response.data);
        } else {
            console.error('Network error:', error.message);
        }
        throw error;
    }
}

// Batch Prediction
async function predictMultiple(assets) {
    try {
        const response = await axios.post(`${BASE_URL}/predict/batch`, {
            assets: assets,
            days_ahead: 1
        });
        
        console.log(`Total predictions: ${response.data.total_assets}`);
        response.data.predictions.forEach(pred => {
            console.log(`${pred.asset}: ${pred.predicted_change_percent}%`);
        });
        
        return response.data;
    } catch (error) {
        console.error('Batch prediction error:', error);
        throw error;
    }
}

// Usage
predictAsset('RELIANCE.NS');
predictMultiple(['RELIANCE.NS', 'TCS.NS', 'GOLD']);
```

---

## 📊 Response Schemas

### PredictionResponse

```json
{
    "asset": "string",
    "current_roi": "float",
    "predicted_roi": "float",
    "predicted_change_percent": "float",
    "confidence": "string (High|Medium|Low)",
    "prediction_date": "string (ISO 8601)",
    "days_ahead": "integer"
}
```

### BatchPredictionResponse

```json
{
    "predictions": ["array of PredictionResponse"],
    "total_assets": "integer",
    "timestamp": "string (ISO 8601)"
}
```

### ErrorResponse

```json
{
    "error": "string",
    "detail": "string (optional)",
    "timestamp": "string (ISO 8601)"
}
```

---

## ❌ Error Codes

### HTTP Status Codes

| Code | Meaning | When it happens |
|------|---------|----------------|
| **200** | OK | Successful request |
| **400** | Bad Request | Invalid parameters (wrong asset, invalid days_ahead) |
| **422** | Unprocessable Entity | Validation error (Pydantic) |
| **500** | Internal Server Error | Server error, prediction failed |
| **503** | Service Unavailable | Model not loaded, data fetch failed |

### Example Error Responses

#### 400 - Invalid Asset
```json
{
    "detail": [
        {
            "loc": ["body", "asset"],
            "msg": "Asset must be one of [...supported assets...]",
            "type": "value_error"
        }
    ]
}
```

#### 400 - Invalid Days Ahead
```json
{
    "detail": [
        {
            "loc": ["body", "days_ahead"],
            "msg": "ensure this value is less than or equal to 30",
            "type": "value_error.number.not_le"
        }
    ]
}
```

#### 503 - Service Unavailable
```json
{
    "error": "Model not loaded",
    "detail": "The prediction model is not available. Please contact support.",
    "timestamp": "2025-11-02T10:30:00.123456"
}
```

---

## 🧪 Test Scenarios

### Positive Tests

1. **Valid Stock Prediction**
   - Asset: "RELIANCE.NS"
   - Expected: 200 OK with prediction

2. **Valid Commodity Prediction**
   - Asset: "GOLD"
   - Expected: 200 OK with prediction

3. **Valid Batch Prediction**
   - Assets: ["RELIANCE.NS", "TCS.NS"]
   - Expected: 200 OK with multiple predictions

4. **Health Check**
   - Expected: 200 OK with status "healthy"

### Negative Tests

1. **Invalid Asset Symbol**
   - Asset: "INVALID.NS"
   - Expected: 400 or 422 with error message

2. **Days Ahead Out of Range**
   - Days ahead: 100
   - Expected: 422 with validation error

3. **Empty Asset List (Batch)**
   - Assets: []
   - Expected: 422 with validation error

4. **Missing Required Fields**
   - Body: {}
   - Expected: 422 with validation error

---

## 📥 Import Postman Collection

Save this as `financial-api.postman_collection.json`:

```json
{
    "info": {
        "name": "Financial Analysis API",
        "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
    },
    "item": [
        {
            "name": "Health Check",
            "request": {
                "method": "GET",
                "url": "{{base_url}}/health"
            }
        },
        {
            "name": "Model Info",
            "request": {
                "method": "GET",
                "url": "{{base_url}}/model/info"
            }
        },
        {
            "name": "Single Prediction",
            "request": {
                "method": "POST",
                "header": [{"key": "Content-Type", "value": "application/json"}],
                "body": {
                    "mode": "raw",
                    "raw": "{\n    \"asset\": \"RELIANCE.NS\",\n    \"days_ahead\": 1\n}"
                },
                "url": "{{base_url}}/predict"
            }
        },
        {
            "name": "Batch Prediction",
            "request": {
                "method": "POST",
                "header": [{"key": "Content-Type", "value": "application/json"}],
                "body": {
                    "mode": "raw",
                    "raw": "{\n    \"assets\": [\"RELIANCE.NS\", \"TCS.NS\", \"GOLD\"],\n    \"days_ahead\": 1\n}"
                },
                "url": "{{base_url}}/predict/batch"
            }
        },
        {
            "name": "Supported Assets",
            "request": {
                "method": "GET",
                "url": "{{base_url}}/assets/supported"
            }
        }
    ],
    "variable": [
        {
            "key": "base_url",
            "value": "http://localhost:8000"
        }
    ]
}
```

**Import to Postman:**
1. Open Postman
2. Click "Import"
3. Paste the JSON above
4. Start testing!

---

**Happy Testing! 🚀**

*For more details, see [API README](README.md)*
