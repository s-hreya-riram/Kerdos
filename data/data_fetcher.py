# data/data_fetcher.py
import yfinance as yf
import pandas as pd
import logging
import time

from .constants import CRYPTO_SYMBOLS
from .data_pipeline import fetch_historical_data

logger = logging.getLogger(__name__)

def fetch_asset_data(symbol_mapping, is_backtesting, start_date=None, end_date=None):
    all_data = {}

    if is_backtesting:
        assert start_date is not None and end_date is not None, \
            "Backtesting requires start_date and end_date"

    else:
        if start_date is None:
            start_date = pd.Timestamp.utcnow() - pd.Timedelta(days=90)
        if end_date is None:
            end_date = pd.Timestamp.utcnow()

    for idx, (alpaca_symbol, yahoo_symbol) in enumerate(symbol_mapping):
        # Use Yahoo Finance for backtesting
        if is_backtesting:
            try:
                if idx > 0:  # Don't wait before first request
                    time.sleep(0.02)  # 20ms delay between each symbol
                ticker = yf.Ticker(yahoo_symbol)
                df = ticker.history(start=start_date, end=end_date)

                if df.empty:
                    raise ValueError("Yahoo returned empty dataframe")

                df.index = pd.to_datetime(df.index).tz_convert("UTC")
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

                logger.info(f"Fetched Yahoo data for {yahoo_symbol} ({len(df)} rows)")
                all_data[yahoo_symbol] = df

            except Exception as e:
                logger.error(f"Yahoo fetch failed for {yahoo_symbol}: {e}")

        # Use Alpaca for live trading / training
        else:
            try:
                is_crypto = CRYPTO_SYMBOLS(alpaca_symbol)

                # passing the feed parameter for Alpaca data fetch
                # as our Alpaca accounts do not have SIP market data permissions for US equities
                # IEX is acceptable for our use case given we're only trading once a day
                bars = fetch_historical_data(
                    symbols=[alpaca_symbol],
                    start_date=start_date,
                    end_date=end_date,
                    feed=None if is_crypto else "iex"
                )

                df = bars[alpaca_symbol].copy()
                df.columns = [c.lower() for c in df.columns]

                if "v" in df.columns and "volume" not in df.columns:
                    df["volume"] = df["v"]

                df = df[['open', 'high', 'low', 'close', 'volume']]
                df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

                df.index = pd.to_datetime(df.index).tz_convert("UTC")

                logger.info(f"Fetched Alpaca data for {alpaca_symbol} ({len(df)} rows)")
                all_data[alpaca_symbol] = df

            except Exception as e:
                logger.error(f"Alpaca fetch failed for {alpaca_symbol}: {e}")

    return all_data

if __name__ == "__main__":
    # Test fetching data for BTC/USD and SPY in backtesting mode
    symbol_mapping = [("BTC/USD", "BTC-USD"), ("SPY", "SPY")]
    start_date = pd.Timestamp("2023-01-01", tz="UTC")
    end_date = pd.Timestamp("2023-06-01", tz="UTC")

    print("Fetching asset data for backtesting...")
    data = fetch_asset_data(symbol_mapping, is_backtesting=True, start_date=start_date, end_date=end_date)
    for symbol, df in data.items():
        print(f"{symbol} data:\n{df.head()}\n")
    
    print("Fetching asset data for live trading...")
    data = fetch_asset_data(symbol_mapping, is_backtesting=False)
    for symbol, df in data.items():
        print(f"{symbol} data:\n{df.head()}\n")