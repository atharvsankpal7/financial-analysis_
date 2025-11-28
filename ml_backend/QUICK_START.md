# 🚀 Quick Start Guide - Financial Analysis ML Backend

Get up and running in 10 minutes!

---

## 📋 Prerequisites Checklist

- [ ] Python 3.9 or higher installed
- [ ] Internet connection (for data fetching)
- [ ] 4GB+ RAM
- [ ] 2GB free disk space

---

## ⚡ 3-Step Quick Start

### Step 1: Install Dependencies (2 minutes)

```powershell
# Navigate to API directory
cd ml_backend\api

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install requirements
pip install -r requirements.txt
```

### Step 2: Verify Model Exists (30 seconds)

```powershell
# Check if model file exists
ls ..\train3\models\hybrid_lstm_attention.keras

# If not found, you need to train the model first
# See: ml_backend/train3/TRAINING_README.md
```

### Step 3: Run API (10 seconds)

```powershell
# Start the server
python main.py

# API will be available at: http://localhost:8000
```

---

## ✅ Test Your Installation

### 1. Open API Documentation

Visit in your browser:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### 2. Test Health Check

```powershell
# Using curl
curl http://localhost:8000/health

# Expected response:
# {"status":"healthy","model_loaded":true,"timestamp":"...","version":"1.0.0"}
```

### 3. Make Your First Prediction

```powershell
# Using curl
curl -X POST "http://localhost:8000/predict" `
  -H "Content-Type: application/json" `
  -d '{"asset": "RELIANCE.NS", "days_ahead": 1}'

# Expected response:
# {
#   "asset": "RELIANCE.NS",
#   "current_roi": 1.25,
#   "predicted_roi": 1.48,
#   "predicted_change_percent": 18.4,
#   "confidence": "High",
#   ...
# }
```

### 4. Run Example Scripts

```powershell
python examples.py
```

---

## 🔧 Common Quick Fixes

### "Model not found" Error

```powershell
# Train the model first
cd ..\train3
python train.py
# Wait 30-60 minutes for training to complete
```

### "Module not found" Error

```powershell
# Make sure virtual environment is activated
.\venv\Scripts\activate

# Reinstall requirements
pip install -r requirements.txt
```

### "Port already in use" Error

```powershell
# Use a different port
uvicorn main:app --port 8001

# Or kill the process using port 8000
# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

---

## 📊 Complete Workflow (If Starting from Scratch)

### 1. Collect Data (15 minutes)

```powershell
cd ml_backend\data_scapping
python fetch_nifty50_data.py
# Output: dataset in data/ folder
```

### 2. Train Model (30-60 minutes)

```powershell
cd ..\train3
python train.py
# Output: models/hybrid_lstm_attention.keras
```

### 3. Run API (instant)

```powershell
cd ..\api
python main.py
# Output: API running on http://localhost:8000
```

---

## 🎯 What's Next?

After getting the API running:

1. **Explore the API**: http://localhost:8000/docs
2. **Read Full Documentation**: 
   - [API Guide](README.md)
   - [Training Guide](../train3/TRAINING_README.md)
   - [Data Collection Guide](../data_scapping/README.md)
3. **Integrate with Frontend**: Use the API endpoints in your application
4. **Customize**: Add more stocks, adjust model parameters
5. **Deploy**: Deploy to cloud (AWS, Heroku, etc.)

---

## 📚 Key Endpoints to Try

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Check API status |
| `/model/info` | GET | Get model details |
| `/predict` | POST | Single prediction |
| `/predict/batch` | POST | Multiple predictions |
| `/assets/supported` | GET | List all assets |

---

## 🐛 Still Having Issues?

1. Check the logs in the console
2. Review the [Troubleshooting](README.md#-troubleshooting) section
3. Open an issue on GitHub
4. Check that all paths in `main.py` Config class are correct

---

## 🎓 Learning Resources

- **FastAPI Tutorial**: https://fastapi.tiangolo.com/tutorial/
- **TensorFlow Guide**: https://www.tensorflow.org/guide
- **yfinance Docs**: https://pypi.org/project/yfinance/

---

**🚀 You're ready to go! Happy predicting!**

*For detailed documentation, see [README.md](README.md)*
