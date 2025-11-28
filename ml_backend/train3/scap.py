import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# --- CONFIG ---
stocks = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "LT.NS", "SBIN.NS", "BHARTIARTL.NS", "HINDUNILVR.NS", "KOTAKBANK.NS",
    "ITC.NS", "BAJFINANCE.NS", "ADANIENT.NS", "ASIANPAINT.NS", "MARUTI.NS",
    "WIPRO.NS", "HCLTECH.NS", "AXISBANK.NS", "SUNPHARMA.NS", "TITAN.NS"
]

commodities = {
    "Gold": "GC=F",
    "Silver": "SI=F",
    "Platinum": "PL=F"
}

start_date = (datetime.now() - timedelta(days=int(2.5 * 365))).strftime("%Y-%m-%d")
end_date = datetime.now().strftime("%Y-%m-%d")

dfs = []

def get_price(symbol, name=None):
    """Fetch single price series (Adj Close or Close) and flatten MultiIndex columns properly."""
    try:
        df = yf.download(symbol, start=start_date, end=end_date, progress=False)
        # Handle MultiIndex columns like ('Adj Close', 'RELIANCE.NS')
        if isinstance(df.columns, pd.MultiIndex):
            # Prefer Adj Close, else Close
            if 'Adj Close' in df.columns.get_level_values(0):
                sub_df = df['Adj Close']
            else:
                sub_df = df['Close']
            if isinstance(sub_df, pd.DataFrame) and symbol in sub_df.columns:
                sub_df = sub_df[[symbol]]
            sub_df.columns = [name or symbol]
        else:
            # Flat columns
            col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
            sub_df = df[[col]].rename(columns={col: name or symbol})
        return sub_df
    except Exception as e:
        print(f"⚠️ Failed {symbol}: {e}")
        return pd.DataFrame()

# --- STOCKS ---
for s in stocks:
    print(f"Fetching {s} ...")
    df = get_price(s)
    if not df.empty:
        dfs.append(df)

# --- COMMODITIES ---
for name, symbol in commodities.items():
    print(f"Fetching {name} ({symbol}) ...")
    df = get_price(symbol, name)
    if not df.empty:
        dfs.append(df)

# --- FD RATE (synthetic example for now) ---
dates = pd.date_range(start=start_date, end=end_date, freq="B")
fd_rate = pd.Series(5.5 + np.random.normal(0, 0.1, len(dates)), index=dates, name="FD_Rate")
dfs.append(fd_rate.to_frame())

# --- MERGE ALL ---
combined = pd.concat(dfs, axis=1).ffill().bfill()

# --- ADD DAILY ROI ---
returns = combined.pct_change().fillna(0)
returns.columns = [str(c) + "_ROI" for c in returns.columns]

final = pd.concat([combined, returns], axis=1)

final.to_csv("dataset_combined_2_5yr.csv")
print(f"✅ Done — saved dataset_combined_2_5yr.csv, shape = {final.shape}")
