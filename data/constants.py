from zoneinfo import ZoneInfo

# Assets to trade
ASSETS = ['SPY', 'GLD', 'VXX', 'BTC/USD']

# Date ranges
START_DATE = '2021-01-31'
END_DATE = '2026-01-31'

# Backtest settings
INITIAL_CAPITAL = 100000
REBALANCE_FREQUENCY = 'daily'

# Set the local timezone
NY_TZ = ZoneInfo('America/New_York')

CRYPTO_SYMBOLS = lambda s: "/" in s or s.endswith("USD")

