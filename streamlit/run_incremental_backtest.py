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
from strategies.xgboost_strategy import MLPortfolioStrategy
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

def run_incremental_backtest(strategy_name, strategy_class, uploader, backtest_start=datetime(2024, 1, 1), backtest_end=datetime.now() - timedelta(days=1)):
    """
    Run FULL backtest from beginning, but only log NEW dates
    
    Key insight:
    - Always backtest from 2024-01-01 to today
    - This ensures positions carry forward correctly
    - But only log dates we don't already have
    """
    
    print(f"\n{'='*60}")
    print(f"PROCESSING {strategy_name}")
    print(f"{'='*60}")
    
    # Get last date in Snowflake
    last_date, _ = get_last_date_in_snowflake(strategy_name)
    
    # Define backtest range
    backtest_start = datetime(2024, 1, 1)
    backtest_end = datetime.now() - timedelta(days=1)  # Yesterday
    
    # Skip weekends
    while backtest_end.weekday() >= 5:
        backtest_end -= timedelta(days=1)
    
    if last_date:
        print(f"📊 Last data in DB: {last_date.date()}")
        print(f"📈 Will backtest: {backtest_start.date()} → {backtest_end.date()}")
        print(f"💾 Will only LOG data after: {last_date.date()}")
        
        # Check if up-to-date
        if last_date.date() >= backtest_end.date():
            print(f"✅ {strategy_name} already up-to-date!")
            return True
    else:
        print(f"⚠️  No existing data - running FULL backtest")
        print(f"📈 Will backtest: {backtest_start.date()} → {backtest_end.date()}")
        print(f"💾 Will log ALL data")
    
    # Create callback with date filter
    callback = create_callback_for_strategy(
        uploader, 
        strategy_name, 
        last_date  # ← Pass last date to filter
    )
    
    try:
        # ========== RUN FULL BACKTEST ==========
        # This ensures positions carry forward correctly
        result = strategy_class.run_backtest(
            YahooDataBacktesting,
            backtest_start,  # Always from beginning
            backtest_end,
            budget=10000,  # Always start with $10k (positions will build up)
            show_plot=False,
            save_tearsheet=False,
            parameters={
                "performance_callback": callback
            }
        )
        # ========== END BACKTEST ==========
        
        print(f"✅ {strategy_name} backtest complete")
        return True
        
    except Exception as e:
        print(f"❌ {strategy_name} failed: {e}")
        import traceback
        traceback.print_exc()
        return False


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