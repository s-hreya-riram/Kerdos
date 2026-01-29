"""
Simple data fetcher for RL training
"""
import yfinance as yf
import pandas as pd
import numpy as np


def get_data(start_date, end_date=None):
    """
    Fetch BTC data and calculate basic features
    
    Returns:
        DataFrame with price and features
    """
    print(f"Fetching BTC data from {start_date} to {end_date}...")

    # Download BTC
    # TODO extend this to include other assets of interest
    btc = yf.download("BTC-USD", start=start_date, end=end_date, progress=False)

    # Flatten MultiIndex columns
    btc.columns = btc.columns.get_level_values(0)
    
    # Calculate returns
    btc['returns'] = btc['Close'].pct_change()
    
    # Simple moving averages
    btc['ma_20'] = btc['Close'].rolling(20).mean()
    btc['ma_50'] = btc['Close'].rolling(50).mean()
    
    # Price relative to MA
    btc['price_to_ma20'] = btc['Close'] / btc['ma_20'] - 1
    
    # Volatility
    btc['volatility'] = btc['returns'].rolling(20).std()
    
    # Drop NaN
    btc = btc.dropna()
    
    print(f"Got {len(btc)} days of data")
    
    return btc

if __name__ == "__main__":
    data = get_data("2020-01-01")
    print(data.tail())