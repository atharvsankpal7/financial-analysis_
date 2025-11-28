import os
import time
import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm
from datetime import datetime, timedelta

# ==============================
# CONFIG
# ==============================
OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Last 6 months
END = datetime.today()
START = END - timedelta(days=180)

# NIFTY 50 Stocks (as per NSE listing)
NIFTY50_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS",
    "BAJFINANCE.NS", "LT.NS", "ASIANPAINT.NS", "MARUTI.NS", "AXISBANK.NS",
    "SUNPHARMA.NS", "ULTRACEMCO.NS", "TITAN.NS", "WIPRO.NS", "ONGC.NS",
]

# Commodities (Yahoo tickers)
COMMODITY_TICKERS = {
    "GOLD": "GC=F",          # Gold Futures
    "SILVER": "SI=F",        # Silver Futures
    "PLATINUM": "PL=F"       # Platinum Futures
}

# ==============================
# HELPERS
# ==============================

def fetch_ticker_data(ticker, label):
    """Fetch and return 6 months of OHLC data for a ticker."""
    try:
        df = yf.download(ticker, start=START, end=END, progress=False, interval="1d")
        if df.empty:
            print(f"⚠️ No data for {label}")
            return None
        df.reset_index(inplace=True)
        return df
    except Exception as e:
        print(f"❌ Failed to fetch {label}: {e}")
        return None


def clean_df(df, label):
    """Standardize dataframe columns."""
    # Handle multi-index columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    wanted = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    for col in wanted:
        if col not in df.columns:
            df[col] = np.nan

    # Force consistent dtypes
    df["Date"] = pd.to_datetime(df["Date"])
    df["Symbol"] = label
    return df[wanted + ["Symbol"]]


def fetch_fd_rate():
    """Mock FD rate dataset (since there’s no public API for it)."""
    # Assume ~7% FD rate with slight daily fluctuation
    dates = pd.date_range(start=START, end=END, freq="D")
    rates = 7 + np.random.normal(0, 0.05, len(dates))
    df = pd.DataFrame({
        "Date": dates,
        "Open": rates,
        "High": rates,
        "Low": rates,
        "Close": rates,
        "Adj Close": rates,
        "Volume": 0
    })
    df["Symbol"] = "FD_RATE"
    return df


# ==============================
# SCRAPING
# ==============================

all_data = []

# --- Stocks ---
for ticker in tqdm(NIFTY50_TICKERS, desc="Fetching NIFTY50 Stocks"):
    label = ticker.replace(".NS", "")
    df = fetch_ticker_data(ticker, label)
    if df is not None:
        df = clean_df(df, label)
        df.to_csv(os.path.join(OUTPUT_DIR, f"{label}.csv"), index=False)
        all_data.append(df)
    time.sleep(0.5)

# --- Commodities ---
for label, ticker in tqdm(COMMODITY_TICKERS.items(), desc="Fetching Commodities"):
    df = fetch_ticker_data(ticker, label)
    if df is not None:
        df = clean_df(df, label)
        df.to_csv(os.path.join(OUTPUT_DIR, f"{label}.csv"), index=False)
        all_data.append(df)
    time.sleep(0.5)

# --- FD Rate ---
fd_df = fetch_fd_rate()
fd_df.to_csv(os.path.join(OUTPUT_DIR, "FD_RATE.csv"), index=False)
all_data.append(fd_df)

# ==============================
# MERGE AND CLEAN
# ==============================

if all_data:
    combined = pd.concat(all_data, ignore_index=True)
    combined.drop_duplicates(subset=["Date", "Symbol"], inplace=True)
    combined.sort_values(by=["Symbol", "Date"], inplace=True)
    final_path = os.path.join(OUTPUT_DIR, "all_assets_6months.csv")
    combined.to_csv(final_path, index=False)
    print(f"\n✅ Combined dataset saved to: {final_path}")
    print(f"Rows: {len(combined)}, Symbols: {combined['Symbol'].nunique()}")
else:
    print("❌ No data fetched.")
