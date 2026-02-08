# scripts/daily_snapshot.py

"""
GitHub Actions daily job - captures current portfolio state and logs to Snowflake
"""

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from lumibot.brokers import Alpaca
from config import ALPACA_CONFIG
from snowflake_logger import SnowflakePerformanceLogger

def daily_snapshot():
    """Capture current portfolio state and log to Snowflake"""
    
    logger = SnowflakePerformanceLogger()
    broker = Alpaca(ALPACA_CONFIG)
    
    # Get account info
    account = broker.get_account()
    positions = broker.get_positions()
    
    # Build snapshot
    positions_dict = {}
    for pos in positions:
        positions_dict[pos.symbol] = {
            'quantity': float(pos.qty),
            'value': float(pos.market_value),
            'avg_price': float(pos.avg_entry_price)
        }
    
    snapshot = {
        'timestamp': datetime.now(),
        'portfolio_value': float(account.portfolio_value),
        'cash': float(account.cash),
        'positions': positions_dict,
        'is_out_of_sample': True,  # We're in live trading
        'weights': {},  # Can be calculated from positions
        'predictions': {}  # Not available in snapshot
    }
    
    # Log to Snowflake
    logger.log_performance(snapshot)
    logger.close()
    
    print(f"✅ Daily snapshot complete: ${snapshot['portfolio_value']:,.2f}")

if __name__ == "__main__":
    daily_snapshot()