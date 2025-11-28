# 📊 Data Collection & Scraping Guide

Comprehensive documentation for collecting financial data for the prediction model.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![yfinance](https://img.shields.io/badge/yfinance-0.2.32-green.svg)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Quick Start](#-quick-start)
- [Data Sources](#-data-sources)
- [Data Collection Process](#-data-collection-process)
- [Data Structure](#-data-structure)
- [Configuration](#-configuration)
- [Output Files](#-output-files)
- [Troubleshooting](#-troubleshooting)
- [Best Practices](#-best-practices)

---

## 🎯 Overview

### What Does This Script Do?

The data collection scripts fetch historical financial data for:
- **Nifty 50 Stocks**: Top Indian stock market companies
- **Commodities**: Gold, Silver, Platinum
- **Fixed Income**: FD (Fixed Deposit) rates

### Key Features

- ✅ **Automated Data Fetching**: Downloads data from Yahoo Finance
- ✅ **Multi-Asset Support**: Stocks, commodities, and fixed income
- ✅ **ROI Calculation**: Automatically calculates Return on Investment
- ✅ **Data Cleaning**: Handles missing values and outliers
- ✅ **CSV Export**: Saves data in structured CSV format
- ✅ **Error Handling**: Robust error handling for failed downloads

---

## 🚀 Quick Start

### Prerequisites

```bash
# Ensure Python 3.9+ is installed
python --version

# Install required packages
pip install yfinance pandas numpy
```

### Run Data Collection

```bash
# Navigate to data scraping directory
cd ml_backend/data_scapping

# Run the script
python fetch_nifty50_data.py
```

### What Happens

1. Script fetches 6 months of historical data
2. Calculates ROI for each asset
3. Saves individual CSV files for each asset
4. Creates a combined dataset `all_assets_6months.csv`

---

## 📡 Data Sources

### Yahoo Finance

We use the `yfinance` library to fetch data from Yahoo Finance.

**Advantages:**
- Free and reliable
- Real-time and historical data
- Global market coverage
- No API key required

**Limitations:**
- Rate limiting (avoid too many rapid requests)
- Market hours only for some data
- Occasional service interruptions

### Supported Assets

#### Nifty 50 Stocks (20 Selected)

| Symbol | Company Name | Sector |
|--------|-------------|--------|
| RELIANCE.NS | Reliance Industries | Energy |
| TCS.NS | Tata Consultancy Services | IT |
| INFY.NS | Infosys | IT |
| HDFCBANK.NS | HDFC Bank | Banking |
| ICICIBANK.NS | ICICI Bank | Banking |
| LT.NS | Larsen & Toubro | Infrastructure |
| SBIN.NS | State Bank of India | Banking |
| BHARTIARTL.NS | Bharti Airtel | Telecom |
| HINDUNILVR.NS | Hindustan Unilever | FMCG |
| KOTAKBANK.NS | Kotak Mahindra Bank | Banking |
| ITC.NS | ITC Limited | FMCG |
| BAJFINANCE.NS | Bajaj Finance | Finance |
| ASIANPAINT.NS | Asian Paints | Paints |
| MARUTI.NS | Maruti Suzuki | Automotive |
| WIPRO.NS | Wipro | IT |
| AXISBANK.NS | Axis Bank | Banking |
| SUNPHARMA.NS | Sun Pharma | Pharma |
| TITAN.NS | Titan Company | Jewelry |
| ONGC.NS | ONGC | Oil & Gas |
| ULTRACEMCO.NS | UltraTech Cement | Cement |

#### Commodities

| Name | Symbol | Description |
|------|--------|-------------|
| Gold | GC=F | Gold Futures |
| Silver | SI=F | Silver Futures |
| Platinum | PL=F | Platinum Futures |

---

## 🔄 Data Collection Process

### Script Workflow

```
┌─────────────────────────────────────┐
│   1. Initialize Configuration       │
│   - Set date range                  │
│   - Define asset lists              │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│   2. Fetch Stock Data               │
│   - Loop through stock symbols      │
│   - Download from Yahoo Finance     │
│   - Extract Adj Close prices        │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│   3. Fetch Commodity Data           │
│   - Download Gold, Silver, Platinum │
│   - Handle MultiIndex columns       │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│   4. Calculate ROI                  │
│   ROI = (Price_t - Price_t-1) /     │
│         Price_t-1 * 100             │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│   5. Handle Missing Data            │
│   - Replace inf/-inf with NaN       │
│   - Forward fill missing values     │
│   - Backward fill remaining gaps    │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│   6. Save Individual CSVs           │
│   - One file per asset              │
│   - Contains price & ROI            │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│   7. Combine All Assets             │
│   - Merge on date index             │
│   - Create unified dataset          │
│   - Save as all_assets_6months.csv  │
└─────────────────────────────────────┘
```

### Code Example: Fetching Single Asset

```python
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def fetch_stock_data(symbol, start_date, end_date):
    """
    Fetch stock data from Yahoo Finance
    
    Args:
        symbol: Stock symbol (e.g., 'RELIANCE.NS')
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
    
    Returns:
        DataFrame with price data
    """
    try:
        # Download data
        df = yf.download(
            symbol, 
            start=start_date, 
            end=end_date, 
            progress=False
        )
        
        # Handle MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            # Prefer Adj Close, else Close
            if 'Adj Close' in df.columns.get_level_values(0):
                df = df['Adj Close']
            else:
                df = df['Close']
        else:
            # Flat columns
            col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
            df = df[[col]]
        
        # Rename column
        df.columns = [symbol]
        
        return df
        
    except Exception as e:
        print(f"⚠️ Failed to fetch {symbol}: {e}")
        return pd.DataFrame()
```

### ROI Calculation

```python
def calculate_roi(prices):
    """
    Calculate Return on Investment (ROI)
    
    Formula: ROI = (Price_current - Price_previous) / Price_previous * 100
    
    Args:
        prices: Series or DataFrame with price data
    
    Returns:
        Series with ROI percentages
    """
    if isinstance(prices, pd.DataFrame):
        prices = prices.iloc[:, 0]
    
    # Calculate percentage change
    roi = prices.pct_change() * 100
    
    # Fill first NaN with 0
    roi = roi.fillna(0)
    
    return roi
```

### Data Cleaning

```python
def clean_data(df):
    """
    Clean financial data
    
    Steps:
    1. Replace infinite values with NaN
    2. Forward fill missing values
    3. Backward fill remaining missing values
    
    Args:
        df: DataFrame to clean
    
    Returns:
        Cleaned DataFrame
    """
    import numpy as np
    
    # Replace inf/-inf with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Forward fill
    df = df.ffill()
    
    # Backward fill
    df = df.bfill()
    
    # If still NaN, fill with 0
    df = df.fillna(0)
    
    return df
```

---

## 📁 Data Structure

### Individual Asset Files

**File**: `RELIANCE.csv`

| Date | RELIANCE.NS | ROI_RELIANCE.NS |
|------|-------------|-----------------|
| 2025-05-02 | 2458.30 | 0.00 |
| 2025-05-03 | 2475.60 | 0.70 |
| 2025-05-06 | 2462.15 | -0.54 |
| 2025-05-07 | 2489.90 | 1.13 |

**Columns:**
- `Date`: Trading date (index)
- `{SYMBOL}`: Adjusted close price
- `ROI_{SYMBOL}`: Daily ROI percentage

### Combined Dataset

**File**: `all_assets_6months.csv`

| Date | RELIANCE.NS | ROI_RELIANCE.NS | TCS.NS | ROI_TCS.NS | GOLD | ROI_GOLD | ... |
|------|-------------|-----------------|--------|------------|------|----------|-----|
| 2025-05-02 | 2458.30 | 0.00 | 3856.45 | 0.00 | 2320.50 | 0.00 | ... |
| 2025-05-03 | 2475.60 | 0.70 | 3871.20 | 0.38 | 2318.75 | -0.08 | ... |

**Structure:**
- One row per trading day
- Columns alternating: Price, ROI, Price, ROI, ...
- Date as index
- All assets aligned by date

---

## ⚙️ Configuration

### Time Period Configuration

```python
# In scap.py or fetch_nifty50_data.py

from datetime import datetime, timedelta

# Last 6 months
start_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
end_date = datetime.now().strftime("%Y-%m-%d")

# Or specific dates
start_date = "2023-01-01"
end_date = "2025-11-02"

# For training (2.5 years)
start_date = (datetime.now() - timedelta(days=int(2.5 * 365))).strftime("%Y-%m-%d")
```

### Adding New Assets

```python
# Add stocks to the list
stocks = [
    "RELIANCE.NS",
    "TCS.NS",
    # Add more...
    "NEWSTOCK.NS",  # Your new stock
]

# Add commodities
commodities = {
    "Gold": "GC=F",
    "Silver": "SI=F",
    "Platinum": "PL=F",
    "Oil": "CL=F",  # Your new commodity
}
```

### Customizing Output

```python
# Change output directory
output_dir = "data/historical"
os.makedirs(output_dir, exist_ok=True)

# Change file naming
filename = f"{symbol}_{start_date}_{end_date}.csv"

# Add more columns
df['Price_MA7'] = df[symbol].rolling(7).mean()  # 7-day moving average
df['Volatility'] = df['ROI'].rolling(30).std()  # 30-day volatility
```

---

## 📤 Output Files

### Generated Files

After running the script, you'll find:

```
data_scapping/
├── fetch_nifty50_data.py        # Main script
├── scap.py                      # Alternative script
├── req.txt                      # Requirements
└── data/
    ├── all_assets_6months.csv   # ✅ Combined dataset
    ├── RELIANCE.csv             # Individual files
    ├── TCS.csv
    ├── INFY.csv
    ├── HDFCBANK.csv
    ├── GOLD.csv
    ├── SILVER.csv
    └── ...
```

### File Usage

| File | Used By | Purpose |
|------|---------|---------|
| `all_assets_6months.csv` | API Server | Real-time predictions |
| `dataset_combined_2_5yr.csv` | Training | Model training |
| Individual CSVs | Analysis | Asset-specific analysis |

---

## 🐛 Troubleshooting

### Common Issues

#### 1. Yahoo Finance Connection Error

**Error**: `No data found, symbol may be delisted`

**Solutions:**
```python
# Verify symbol format
# NSE stocks: Add .NS suffix
"RELIANCE.NS" ✅
"RELIANCE"    ❌

# Check if stock is still listed
# Try on Yahoo Finance website first

# Add retry logic
import time

for attempt in range(3):
    try:
        df = yf.download(symbol, ...)
        break
    except Exception as e:
        if attempt < 2:
            time.sleep(5)
            continue
        else:
            print(f"Failed after 3 attempts: {e}")
```

#### 2. Rate Limiting

**Error**: `Too many requests`

**Solution:**
```python
import time

for symbol in stocks:
    df = fetch_stock_data(symbol, ...)
    time.sleep(1)  # Add 1 second delay between requests
```

#### 3. Missing Data / NaN Values

**Symptoms**: Gaps in data, NaN values

**Solutions:**
```python
# Check data completeness
print(df.isnull().sum())

# Fill missing values
df = df.ffill().bfill()

# Interpolate missing values
df = df.interpolate(method='linear')

# Drop incomplete rows (use with caution)
df = df.dropna()
```

#### 4. MultiIndex Column Issues

**Error**: `KeyError` when accessing columns

**Solution:**
```python
# Check column structure
print(df.columns)
print(df.columns.levels if isinstance(df.columns, pd.MultiIndex) else "Flat")

# Flatten MultiIndex
if isinstance(df.columns, pd.MultiIndex):
    df.columns = ['_'.join(col).strip() for col in df.columns.values]
```

#### 5. Date Format Issues

**Error**: Date parsing errors

**Solution:**
```python
# Ensure date format
start_date = datetime.strptime("2025-01-01", "%Y-%m-%d").strftime("%Y-%m-%d")

# Parse dates when reading CSV
df = pd.read_csv('data.csv', parse_dates=['Date'], index_col='Date')

# Convert string to datetime
df.index = pd.to_datetime(df.index)
```

---

## 📊 Data Quality Checks

### Validation Script

```python
def validate_data(df, asset_name):
    """
    Validate financial data quality
    
    Checks:
    1. No missing dates (weekends/holidays okay)
    2. No extreme outliers in ROI
    3. Sufficient data points
    4. No duplicate dates
    """
    issues = []
    
    # Check for duplicates
    if df.index.duplicated().any():
        issues.append(f"❌ Duplicate dates found")
    
    # Check data length
    if len(df) < 100:
        issues.append(f"⚠️ Only {len(df)} data points (need 100+)")
    
    # Check for extreme ROI values
    roi_col = f"ROI_{asset_name}"
    if roi_col in df.columns:
        extreme = df[df[roi_col].abs() > 20]  # More than 20% change
        if not extreme.empty:
            issues.append(f"⚠️ {len(extreme)} days with extreme ROI (>20%)")
    
    # Check for missing values
    missing = df.isnull().sum().sum()
    if missing > 0:
        issues.append(f"⚠️ {missing} missing values")
    
    # Report
    if issues:
        print(f"\n{asset_name} Data Quality Issues:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print(f"✅ {asset_name}: Data quality good")
    
    return len(issues) == 0

# Use after collecting data
for asset in assets:
    df = pd.read_csv(f'data/{asset}.csv', index_col=0, parse_dates=True)
    validate_data(df, asset)
```

---

## 🎯 Best Practices

### 1. Schedule Regular Updates

```bash
# Use cron (Linux/Mac) or Task Scheduler (Windows)
# Run daily at 8 PM after market close

# crontab entry
0 20 * * 1-5 cd /path/to/project && python fetch_nifty50_data.py
```

### 2. Version Your Data

```python
from datetime import datetime

# Add timestamp to filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"all_assets_6months_{timestamp}.csv"
```

### 3. Backup Important Data

```bash
# Create backups directory
mkdir -p data/backups

# Copy before overwriting
cp data/all_assets_6months.csv data/backups/all_assets_$(date +%Y%m%d).csv
```

### 4. Log Data Collection

```python
import logging

logging.basicConfig(
    filename='data_collection.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logging.info(f"Started data collection for {len(stocks)} stocks")
logging.info(f"Date range: {start_date} to {end_date}")
```

### 5. Monitor Data Freshness

```python
def check_data_freshness(csv_file, max_age_days=2):
    """Check if data is recent enough"""
    df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
    last_date = df.index[-1]
    age = (datetime.now() - last_date).days
    
    if age > max_age_days:
        print(f"⚠️ Data is {age} days old. Consider updating.")
    else:
        print(f"✅ Data is fresh ({age} days old)")
```

---

## 📈 Advanced Features

### Parallel Data Fetching

```python
from concurrent.futures import ThreadPoolExecutor
import threading

def fetch_parallel(symbols, start_date, end_date, max_workers=5):
    """Fetch multiple symbols in parallel"""
    results = {}
    lock = threading.Lock()
    
    def fetch_and_store(symbol):
        df = fetch_stock_data(symbol, start_date, end_date)
        with lock:
            results[symbol] = df
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(fetch_and_store, symbols)
    
    return results

# Use
results = fetch_parallel(stocks, start_date, end_date)
```

### Add Technical Indicators

```python
def add_technical_indicators(df, price_col):
    """Add common technical indicators"""
    
    # Simple Moving Averages
    df['SMA_7'] = df[price_col].rolling(7).mean()
    df['SMA_30'] = df[price_col].rolling(30).mean()
    
    # Exponential Moving Average
    df['EMA_12'] = df[price_col].ewm(span=12).mean()
    
    # Bollinger Bands
    df['BB_middle'] = df[price_col].rolling(20).mean()
    df['BB_std'] = df[price_col].rolling(20).std()
    df['BB_upper'] = df['BB_middle'] + (df['BB_std'] * 2)
    df['BB_lower'] = df['BB_middle'] - (df['BB_std'] * 2)
    
    # RSI (Relative Strength Index)
    delta = df[price_col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df
```

---

## 📚 References

- [yfinance Documentation](https://python-yahoofinance.readthedocs.io/)
- [Yahoo Finance](https://finance.yahoo.com/)
- [NSE India](https://www.nseindia.com/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)

---

## ✅ Data Collection Checklist

Before running:
- [ ] Python 3.9+ installed
- [ ] Required packages installed
- [ ] Internet connection active
- [ ] Sufficient disk space

After running:
- [ ] All CSV files generated
- [ ] Combined dataset created
- [ ] Data quality checks passed
- [ ] No error messages in console
- [ ] Backup created (optional)

---

**Happy Data Collecting! 📊**

*Last Updated: November 2025*
