import os
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.data.historical.crypto import CryptoHistoricalDataClient
from alpaca.data.historical.stock import StockHistoricalDataClient

load_dotenv()
API_KEY = os.getenv('ALPACA_API_KEY')
SECRET_KEY = os.getenv('ALPACA_API_SECRET')

STOCK_CLIENT = StockHistoricalDataClient(API_KEY, SECRET_KEY)
CRYPTO_CLIENT = CryptoHistoricalDataClient(API_KEY, SECRET_KEY)

TRADING_CLIENT = TradingClient(API_KEY, SECRET_KEY, paper=True)