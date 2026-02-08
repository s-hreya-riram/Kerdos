import os
import streamlit as st

from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.data.historical.crypto import CryptoHistoricalDataClient
from alpaca.data.historical.stock import StockHistoricalDataClient


load_dotenv()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY") or st.secrets.get("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET") or st.secrets.get("ALPACA_API_SECRET")


ALPACA_CONFIG = {
    "API_KEY": ALPACA_API_KEY,
    "API_SECRET": ALPACA_API_SECRET,
    "PAPER": True,
}

STOCK_CLIENT = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)
CRYPTO_CLIENT = CryptoHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)

TRADING_CLIENT = TradingClient(ALPACA_API_KEY, ALPACA_API_SECRET, paper=True)