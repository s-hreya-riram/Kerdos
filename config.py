import os
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.data.historical.crypto import CryptoHistoricalDataClient
from alpaca.data.historical.stock import StockHistoricalDataClient

load_dotenv()
API_KEY = os.getenv('ALPACA_API_KEY')
SECRET_KEY = os.getenv('ALPACA_API_SECRET')

ALPACA_CONFIG = {
    "API_KEY": os.getenv("ALPACA_API_KEY"),
    "API_SECRET": os.getenv("ALPACA_API_SECRET"),
    "PAPER": True,
}

STOCK_CLIENT = StockHistoricalDataClient(API_KEY, SECRET_KEY)
CRYPTO_CLIENT = CryptoHistoricalDataClient(API_KEY, SECRET_KEY)

TRADING_CLIENT = TradingClient(API_KEY, SECRET_KEY, paper=True)