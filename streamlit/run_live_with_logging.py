"""
Live trading script with Snowflake logging.
This file wires together the strategy and the logger.
"""

import os
from datetime import datetime
from lumibot.brokers import Alpaca
from config import ALPACA_CONFIG
from strategies.xgboost_strategy import MLPortfolioStrategy
from snowflake_logger import SnowflakePerformanceLogger

def run_live_trading_with_logging():
    """
    Run live trading with Snowflake logging enabled
    """
    
    print("🚀 Starting live trading with Snowflake logging...")
    
    # 1. Initialize Snowflake logger
    logger = SnowflakePerformanceLogger()
    
    # 2. Initialize broker
    broker = Alpaca(ALPACA_CONFIG)
    
    # 3. Initialize strategy WITH callback
    strategy = MLPortfolioStrategy(
        broker=broker,
        performance_callback=logger.log_performance  # ← Pass logger's method as callback
    )
    
    # 4. Run strategy (live trading)
    try:
        strategy.run()
    except KeyboardInterrupt:
        print("\n⚠️ Stopping strategy...")
    finally:
        logger.close()
        print("✅ Strategy stopped and Snowflake connection closed")

if __name__ == "__main__":
    run_live_trading_with_logging()