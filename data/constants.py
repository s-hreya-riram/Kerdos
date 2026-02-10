from zoneinfo import ZoneInfo

# Assets to trade, maintaining tuples with the Alpaca and Yahoo Finance symbols
# Alpaca is used for training/live trading and Yahoo Finance is used for backtesting
ASSETS = [
    ("BTC/USD", "BTC-USD"),
    #("ETH/USD", "ETH-USD"),
    ("SPY", "SPY"),
    ("GLD", "GLD"),
    ("SLV", "SLV"),
    #("VXX", "VXX"),
    #("VCMDX", "VCMDX"),
    ("SMH", "SMH"),
    ("ZAP", "ZAP")
]

# Date ranges
START_DATE = '2021-01-31'
END_DATE = '2026-01-31'

# Backtest settings
REBALANCE_FREQUENCY = 'daily'

# Set the local timezone
NY_TZ = ZoneInfo('America/New_York')

CRYPTO_SYMBOLS = lambda s: "/" in s or s.endswith("USD")

# using the best hyperparameters found from hyperparameter_tuning.py using Bayesian Optimization
XGB_MODEL_PARAMS = {'n_estimators': 496, 'max_depth': 6, 'learning_rate': 0.006804969902753306, 'subsample': 0.9209500039708808, 'colsample_bytree': 0.9546460881593619, 'min_child_weight': 6, 'gamma': 9.523455933694268e-05}

MAX_GROSS_EXPOSURE = 0.95
MAX_POSITION_PCT = 0.33
MIN_TRADE_DOLLARS = 100
