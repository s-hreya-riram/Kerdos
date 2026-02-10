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
    Get the last date we have data for in Snowflake
    Returns None if no data exists
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
        
        cursor = conn.cursor()
        
        query = f"""
        SELECT MAX(TIMESTAMP) as last_date
        FROM STRATEGY_PERFORMANCE
        WHERE STRATEGY_NAME = '{strategy_name}'
        """
        
        cursor.execute(query)
        result = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        if result and result[0]:
            last_date = pd.to_datetime(result[0])
            print(f"✅ Last {strategy_name} date in Snowflake: {last_date.date()}")
            return last_date
        else:
            print(f"⚠️  No {strategy_name} data found in Snowflake")
            return None
            
    except Exception as e:
        print(f"❌ Error checking Snowflake: {e}")
        return None


def get_next_trading_day(date):
    """
    Get the next trading day after the given date
    Simple version: skip weekends
    """
    next_day = date + timedelta(days=1)
    
    # Skip weekends
    while next_day.weekday() >= 5:  # 5=Saturday, 6=Sunday
        next_day += timedelta(days=1)
    
    return next_day


def get_trading_days_between(start_date, end_date):
    """
    Count trading days between two dates (excluding weekends)
    """
    current = start_date
    count = 0
    
    while current <= end_date:
        if current.weekday() < 5:  # Monday=0, Friday=4
            count += 1
        current += timedelta(days=1)
    
    return count


def run_incremental_backtest(strategy_name, strategy_class, start_date, end_date, uploader):
    """
    Run backtest for a specific date range
    """
    print(f"\n{'='*60}")
    print(f"RUNNING {strategy_name}")
    print(f"{'='*60}")
    print(f"Period: {start_date.date()} to {end_date.date()}")
    
    callback = create_callback_for_strategy(uploader, strategy_name)
    
    try:
        result = strategy_class.run_backtest(
            YahooDataBacktesting,
            start_date,
            end_date,
            budget=10000,  # This should match your initial capital
            show_plot=False,
            save_tearsheet=False,
            parameters={
                "performance_callback": callback
            }
        )
        print(f"✅ {strategy_name} backtest complete")
        return True
        
    except Exception as e:
        print(f"❌ {strategy_name} backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """
    Main incremental backtest runner
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
    
    # Get current date (yesterday, since today's data isn't complete yet)
    today = datetime.now().date()
    yesterday = today - timedelta(days=1)
    end_date = datetime.combine(yesterday, datetime.min.time())
    
    # Skip weekends for end_date
    while end_date.weekday() >= 5:
        end_date -= timedelta(days=1)
    
    print(f"\n📅 Target end date: {end_date.date()}")
    
    # Connect to Snowflake
    print("\n🔗 Connecting to Snowflake...")
    uploader = SnowflakeUploader()
    print("✅ Connected to Snowflake")
    
    # Process each strategy
    for strategy_name, strategy_class in strategies:
        print(f"\n{'='*60}")
        print(f"PROCESSING {strategy_name}")
        print(f"{'='*60}")
        
        # Get last date in Snowflake
        last_date = get_last_date_in_snowflake(strategy_name)
        
        if last_date is None:
            # No data exists - run full backtest from beginning
            print(f"⚠️  No existing data for {strategy_name}")
            print(f"   Running FULL backtest from 2024-01-01")
            start_date = datetime(2024, 1, 1)
        else:
            # Data exists - run incremental backtest
            # Start from next trading day after last date
            start_date = get_next_trading_day(last_date)
            
            print(f"📊 Last data: {last_date.date()}")
            print(f"🔄 Will backtest from: {start_date.date()}")
        
        # Check if we're already up-to-date or need to skip
        if start_date.date() > end_date.date():
            print(f"✅ {strategy_name} is already up-to-date!")
            print(f"   Latest data: {last_date.date() if last_date else 'N/A'}")
            print(f"   Target date: {end_date.date()}")
            continue
        
        # Count trading days between start and end
        trading_days = get_trading_days_between(start_date, end_date)
        calendar_days = (end_date - start_date).days
        
        print(f"📈 Will backtest {trading_days} trading days ({calendar_days} calendar days)")
        
        # CRITICAL: Lumibot needs at least 2 days difference for backtesting
        # If we only have same-day or next-day, we need to extend the range
        if calendar_days < 2:
            # Extend end_date by at least 2 days, skipping weekends
            extended_end = end_date
            days_added = 0
            
            while days_added < 2:
                extended_end += timedelta(days=1)
                if extended_end.weekday() < 5:  # Only count weekdays
                    days_added += 1
            
            print(f"   ⚠️  Single day backtest detected")
            print(f"   📅 Extending end date: {end_date.date()} → {extended_end.date()}")
            print(f"   ℹ️  Note: Lumibot requires multi-day range, extra days won't affect results")
            
            end_date = extended_end
        
        # Run backtest
        success = run_incremental_backtest(
            strategy_name,
            strategy_class,
            start_date,
            end_date,
            uploader
        )
        
        if success:
            print(f"✅ {strategy_name} updated successfully")
        else:
            print(f"❌ {strategy_name} update failed")
    
    # Cleanup
    print("\n🔒 Closing Snowflake connection...")
    uploader.close()
    
    print("\n" + "="*60)
    print("✅ INCREMENTAL BACKTEST COMPLETE")
    print("="*60)
    print(f"End time: {datetime.now()}")
    print("\nData has been updated in Snowflake!")


if __name__ == "__main__":
    main()