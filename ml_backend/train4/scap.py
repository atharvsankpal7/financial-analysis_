# scraper.py
import yfinance as yf
import pandas as pd
import numpy as np
import ta  # pip install ta

tickers = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "LT.NS",
    "SBIN.NS", "BHARTIARTL.NS", "HINDUNILVR.NS", "KOTAKBANK.NS", "ITC.NS",
    "BAJFINANCE.NS", "ADANIENT.NS", "ASIANPAINT.NS", "MARUTI.NS", "WIPRO.NS",
    "HCLTECH.NS", "AXISBANK.NS", "SUNPHARMA.NS", "TITAN.NS",
    "GC=F", "SI=F", "PL=F"  # Gold, Silver, Platinum
]

start_date = "2020-01-01"
end_date = "2025-01-01"

frames = []

print("Fetching data...")

for t in tickers:
    try:
        print(f"→ {t}")
        df = yf.download(t, start=start_date, end=end_date, progress=False, auto_adjust=True)

        if df.empty:
            print(f"⚠️ No data for {t}, skipping.")
            continue

        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Keep only necessary columns
        df = df[['Close', 'Volume']].dropna()

        # Add indicators - ensure we're working with Series
        df['Return'] = df['Close'].pct_change()
        df['SMA_10'] = df['Close'].rolling(10).mean()
        df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
        df['RSI'] = ta.momentum.RSIIndicator(df['Close'].squeeze(), window=14).rsi()
        df['Volatility'] = df['Return'].rolling(10).std()

        df.dropna(inplace=True)

        # Normalize columns per ticker (z-score)
        for col in ['Close', 'Volume', 'Return', 'SMA_10', 'EMA_10', 'RSI', 'Volatility']:
            df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-9)

        df['Ticker'] = t
        frames.append(df.reset_index())

    except Exception as e:
        print(f"❌ Failed {t}: {e}")

if not frames:
    raise ValueError("No valid data fetched.")

final_df = pd.concat(frames, ignore_index=True)
final_df.to_csv("dataset_cleaned.csv", index=False)
print(f"\n✅ Saved dataset_cleaned.csv | Shape = {final_df.shape}")
