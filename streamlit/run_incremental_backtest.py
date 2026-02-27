import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import snowflake.connector
from dotenv import load_dotenv

project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from config import ALPACA_CONFIG
from snowflake_logger import SnowflakeUploader
from strategies.strategy import MLPortfolioStrategy
from strategies.spy_strategy import SPYBenchmarkStrategy
from dual_backtest import create_callback_for_strategy

from lumibot.backtesting import YahooDataBacktesting
from lumibot.brokers import Alpaca

load_dotenv()


def get_last_date_in_snowflake(strategy_name):
    """
    Get the last date and portfolio value we have data for in Snowflake
    Returns (last_date, portfolio_value) or (None, None) if no data exists
    """
    try:
        conn = snowflake.connector.connect(
            user=os.getenv("SNOWFLAKE_USER"),
            password=os.getenv("SNOWFLAKE_PASSWORD"),
            account=os.getenv("SNOWFLAKE_ACCOUNT"),
            warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
            database=os.getenv("SNOWFLAKE_DATABASE"),
            schema=os.getenv("SNOWFLAKE_SCHEMA", "PRODUCTION")
        )
        
        # CRITICAL: Explicitly activate warehouse
        cursor = conn.cursor()
        cursor.execute(f"USE WAREHOUSE {os.getenv('SNOWFLAKE_WAREHOUSE')}")
        
        query = f"""
        SELECT TIMESTAMP, PORTFOLIO_VALUE
        FROM STRATEGY_PERFORMANCE
        WHERE STRATEGY_NAME = '{strategy_name}'
        ORDER BY TIMESTAMP DESC
        LIMIT 1
        """
        
        cursor.execute(query)
        result = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        if result and result[0]:
            last_date = pd.to_datetime(result[0])
            portfolio_value = float(result[1])
            print(f"✅ Last {strategy_name} date in Snowflake: {last_date.date()}")
            print(f"   Portfolio value: ${portfolio_value:,.2f}")
            return last_date, portfolio_value
        else:
            print(f"⚠️  No {strategy_name} data found in Snowflake")
            return None, None
            
    except Exception as e:
        print(f"❌ Error checking Snowflake: {e}")
        return None, None
    
def run_incremental_backtest(strategy_name, strategy_class, uploader):
    """
    Run FULL backtest, then update Snowflake with ALL data
    This ensures returns chain is never broken
    """
    
    print(f"\n{'='*60}")
    print(f"PROCESSING {strategy_name}")
    print(f"{'='*60}")
    
    # Get last date in Snowflake
    last_date, _ = get_last_date_in_snowflake(strategy_name)
    
    # Define backtest range (ALWAYS FULL)
    backtest_start = datetime(2024, 1, 1)
    backtest_end = datetime.now() - timedelta(days=1)
    
    # Skip weekends
    while backtest_end.weekday() >= 5:
        backtest_end -= timedelta(days=1)
    
    if last_date:
        print(f"📊 Last data in DB: {last_date.date()}")
        
        # Check if up-to-date
        if last_date.date() >= backtest_end.date():
            print(f"✅ {strategy_name} already up-to-date!")
            return True
        
        print(f"📈 Running FULL backtest to ensure data continuity")
        # NEW: Delete old data and rewrite (ensures consistency)
        print(f"🗑️  Clearing old {strategy_name} data...")
        delete_strategy_data(strategy_name)
    
    print(f"💾 Will log ALL data from {backtest_start.date()} to {backtest_end.date()}")
    
    # Create callback WITHOUT date filter
    callback = create_callback_for_strategy(
        uploader, 
        strategy_name, 
        None  # ← No date filter, log everything
    )
    
    try:
        result = strategy_class.run_backtest(
            YahooDataBacktesting,
            backtest_start,
            backtest_end,
            budget=10000,
            show_plot=False,
            save_tearsheet=False,
            parameters={
                "performance_callback": callback
            }
        )
        
        print(f"✅ {strategy_name} backtest complete")
        return True
        
    except Exception as e:
        print(f"❌ {strategy_name} failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def delete_strategy_data(strategy_name):
    """Delete existing data for a strategy before rewriting"""
    try:
        conn = snowflake.connector.connect(
            user=os.getenv("SNOWFLAKE_USER"),
            password=os.getenv("SNOWFLAKE_PASSWORD"),
            account=os.getenv("SNOWFLAKE_ACCOUNT"),
            warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
            database=os.getenv("SNOWFLAKE_DATABASE"),
            schema=os.getenv("SNOWFLAKE_SCHEMA", "PRODUCTION")
        )
        
        cursor = conn.cursor()
        cursor.execute(f"USE WAREHOUSE {os.getenv('SNOWFLAKE_WAREHOUSE')}")
        
        # Delete from both tables
        cursor.execute(f"DELETE FROM STRATEGY_PERFORMANCE WHERE STRATEGY_NAME = '{strategy_name}'")
        cursor.execute(f"DELETE FROM STRATEGY_POSITIONS WHERE STRATEGY_NAME = '{strategy_name}'")
        
        cursor.close()
        conn.close()
        
        print(f"✅ Deleted old {strategy_name} data")
        
    except Exception as e:
        print(f"⚠️  Could not delete old data: {e}")


def main():
    """
    Incremental backtest runner
    
    Strategy:
    1. Always run FULL backtest from 2024-01-01
    2. But only LOG new dates to Snowflake
    3. This ensures positions carry forward correctly
    """
    
    print("="*60)
    print("INCREMENTAL BACKTEST RUNNER")
    print("="*60)
    print(f"Run time: {datetime.now()}")
    
    # Define strategies
    strategies = [
        ("ML_XGBOOST", MLPortfolioStrategy),
        ("SPY_BENCHMARK", SPYBenchmarkStrategy)
    ]
    
    # Connect to Snowflake
    print("\n🔗 Connecting to Snowflake...")
    uploader = SnowflakeUploader()
    print("✅ Connected to Snowflake")
    
    # Process each strategy
    for strategy_name, strategy_class in strategies:
        success = run_incremental_backtest(strategy_name, strategy_class, uploader)
        
        if success:
            print(f"✅ {strategy_name} updated successfully")
        else:
            print(f"❌ {strategy_name} update failed")
    
    # Cleanup
    print("\n🔒 Closing Snowflake connection...")
    uploader.close()
    
    print("\n" + "="*60)
    print("✅ INCREMENTAL UPDATE COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()