"""
Run backtest with BOTH ML strategy and SPY benchmark
Logs both to Snowflake for comparison
"""

import os
import sys
from datetime import datetime
from lumibot.backtesting import YahooDataBacktesting
from lumibot.brokers import Alpaca
import pandas as pd
import numpy as np
import math

from snowflake_logger import SnowflakeUploader
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from config import ALPACA_CONFIG
strategies_path = os.path.join(project_root, "strategies")
sys.path.insert(0, strategies_path)
from strategy import MLPortfolioStrategy
from spy_strategy import SPYBenchmarkStrategy
sys.path.pop(0)
data_path = os.path.join(project_root, "data")
sys.path.insert(0, data_path)
from utils import to_utc

def is_valid_number(val):
    """Check if a value is a valid number (not NaN, not inf)"""
    if val is None:
        return False
    if pd.isna(val):
        return False
    if isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            return False
    if isinstance(val, (np.floating,)):
        if np.isnan(val) or np.isinf(val):
            return False
    return True

def create_callback_for_strategy(uploader, strategy_name, last_date_in_db=None):
    """
    Create a callback for logging strategy performance to Snowflake
    
    Args:
        uploader: SnowflakeUploader instance
        strategy_name: Strategy identifier (e.g., 'ML_XGBOOST', 'SPY_BENCHMARK')
        last_date_in_db: Optional - Last date already in Snowflake
                        If None: Logs ALL data (full backtest)
                        If set: Only logs data AFTER this date (incremental)
    
    Usage:
        # Full backtest (log everything):
        callback = create_callback_for_strategy(uploader, "ML_XGBOOST")
        
        # Incremental backtest (only log new dates):
        callback = create_callback_for_strategy(uploader, "ML_XGBOOST", last_date)
    """
    
    def callback(data):
        timestamp = data['timestamp']
        portfolio_value = data['portfolio_value']
        cash = data['cash']
        positions = data['positions']
        is_out_of_sample = data['is_out_of_sample']
        predictions = data.get('predictions', {})
        
        # Convert timestamp to datetime
        if hasattr(timestamp, 'date'):
            current_date = timestamp
        else:
            current_date = pd.to_datetime(timestamp)
        current_date_utc = to_utc(current_date)
        # ========== SMART DATE FILTERING ==========
        if last_date_in_db is not None:
            # Incremental mode: Skip dates we already have
            last_date_in_db_utc = to_utc(last_date_in_db)
            if current_date_utc.date() <= last_date_in_db_utc.date():
                print(f"      ⏭️  Skip {current_date_utc.date()} (already in DB)")
                return
            else:
                print(f"      💾 Logging {current_date_utc.date()} (NEW)")
        else:
            # Full backtest mode: Log everything
            print(f"      💾 Logging {current_date_utc.date()}")
        # ========== END FILTERING ==========
        
        # Prepare performance data
        perf_data = {
            'STRATEGY_NAME': strategy_name,
            'TIMESTAMP': current_date_utc,
            'PORTFOLIO_VALUE': float(portfolio_value),
            'CASH': float(cash),
            'IS_OUT_OF_SAMPLE': bool(is_out_of_sample)
        }
        
        perf_df = pd.DataFrame([perf_data])
        
        # Prepare positions data
        pos_rows = []
        for symbol, pos_info in positions.items():
            pos_rows.append({
                'STRATEGY_NAME': strategy_name,
                'TIMESTAMP': current_date_utc,
                'SYMBOL': str(symbol),
                'QUANTITY': float(pos_info['quantity']),
                'MARKET_VALUE': float(pos_info['value']),
                'AVG_PRICE': float(pos_info.get('avg_price', 0))
            })
        
        pos_df = pd.DataFrame(pos_rows) if pos_rows else pd.DataFrame()
        
        # Prepare predictions data (ML strategies only)
        pred_df = pd.DataFrame()
        if predictions and predictions.get('symbols'):
            pred_rows = []
            for i, symbol in enumerate(predictions['symbols']):
                pred_rows.append({
                    'STRATEGY_NAME': strategy_name,
                    'TIMESTAMP': current_date,
                    'SYMBOL': str(symbol),
                    'PREDICTED_RETURN': float(predictions['returns'][i]) if i < len(predictions['returns']) else None,
                    'PREDICTED_VOL': float(predictions['volatility'][i]) if i < len(predictions['volatility']) else None
                })
            pred_df = pd.DataFrame(pred_rows)
        
        # Upload to Snowflake
        try:
            uploader.upload_performance(perf_df)
            if not pos_df.empty:
                uploader.upload_positions(pos_df)
            if not pred_df.empty:
                uploader.upload_predictions(pred_df)
            
            print(f"         ✅ Uploaded to Snowflake")
        except Exception as e:
            print(f"         ❌ Upload error: {e}")
            import traceback
            traceback.print_exc()
    
    return callback

def run_dual_backtest():
    """
    Run both ML strategy and SPY benchmark, logging both to Snowflake
    """
    
    print("="*60)
    print("DUAL STRATEGY BACKTEST - ROBUST VERSION")
    print("="*60)
    print(f"Start time: {datetime.now()}")
    
    # Backtest parameters
    START_DATE = datetime(2024, 1, 1)
    END_DATE = datetime(2026, 2, 7)
    INITIAL_CAPITAL = 10000
    
    print(f"\nBacktest period: {START_DATE.date()} to {END_DATE.date()}")
    print(f"Initial capital: ${INITIAL_CAPITAL:,}")
    
    # Create Snowflake uploader
    print("\n🔗 Connecting to Snowflake...")
    uploader = SnowflakeUploader()
    print("✅ Connected to Snowflake")
    
    broker = Alpaca(ALPACA_CONFIG)
    
    # ========== RUN ML STRATEGY ==========
    print("\n" + "="*60)
    print("1️⃣  RUNNING ML STRATEGY")
    print("="*60)
    
    ml_callback = create_callback_for_strategy(uploader, "ML_XGBOOST")
    
    try:
        ml_result = MLPortfolioStrategy.run_backtest(
            YahooDataBacktesting,
            START_DATE,
            END_DATE,
            budget=INITIAL_CAPITAL,
            show_plot=False,
            save_tearsheet=False,
            parameters={
                "performance_callback": ml_callback
            }
        )
        print("✅ ML Strategy backtest complete")
        
    except Exception as e:
        print(f"❌ ML Strategy failed: {e}")
        import traceback
        traceback.print_exc()
    
    # ========== RUN SPY BENCHMARK ==========
    print("\n" + "="*60)
    print("2️⃣  RUNNING SPY BENCHMARK")
    print("="*60)
    
    spy_callback = create_callback_for_strategy(uploader, "SPY_BENCHMARK")
    
    try:
        spy_result = SPYBenchmarkStrategy.run_backtest(
            YahooDataBacktesting,
            START_DATE,
            END_DATE,
            budget=INITIAL_CAPITAL,
            show_plot=False,
            save_tearsheet=False,
            parameters={
                "performance_callback": spy_callback
            }
        )
        print("✅ SPY Benchmark backtest complete")
        
    except Exception as e:
        print(f"❌ SPY Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
    
    # ========== CLEANUP ==========
    print("\n🔒 Closing Snowflake connection...")
    uploader.close()
    
    print("\n" + "="*60)
    print("✅ DUAL BACKTEST COMPLETE")
    print("="*60)
    print(f"End time: {datetime.now()}")
    print("\nData has been uploaded to Snowflake:")
    print("  - STRATEGY_PERFORMANCE")
    print("  - STRATEGY_POSITIONS")
    print("  - STRATEGY_PREDICTIONS (with NaN values filtered)")
    print("\nYou can now run the Streamlit dashboard to visualize results!")
    print("\nNote: Some predictions may have been skipped due to NaN values")
    print("      This is expected when ZAP has no data before Dec 2024")


if __name__ == "__main__":
    run_dual_backtest()