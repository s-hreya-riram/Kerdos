from zoneinfo import ZoneInfo

# Assets to trade, maintaining tuples with the Alpaca and Yahoo Finance symbols
# Alpaca is used for training/live trading and Yahoo Finance is used for backtesting
ASSETS = [
    ("BTC/USD", "BTC-USD"),
    ("ETH/USD", "ETH-USD"),
    ("SPY", "SPY"),
    ("GLD", "GLD"),
    ("VXX", "VXX")
]

# Date ranges
START_DATE = '2021-01-31'
END_DATE = '2026-01-31'

# Backtest settings
INITIAL_CAPITAL = 100000
REBALANCE_FREQUENCY = 'daily'

# Set the local timezone
NY_TZ = ZoneInfo('America/New_York')

CRYPTO_SYMBOLS = lambda s: "/" in s or s.endswith("USD")

XGB_MODEL_PARAMS = {
                        "n_estimators": 300,
                        "max_depth": 4,
                        "learning_rate": 0.05,
                        "subsample": 0.8,
                        "colsample_bytree": 0.8,
                        "objective": "reg:squarederror",
                        "random_state": 42,
                        "n_jobs": -1,
                    }

