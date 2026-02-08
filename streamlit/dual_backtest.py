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
from xgboost_strategy import MLPortfolioStrategy
from spy_strategy import SPYBenchmarkStrategy

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


def create_callback_for_strategy(uploader: SnowflakeUploader, strategy_name: str):
    """
    Creates a callback function that logs to Snowflake with the given strategy name
    ROBUST version with complete NaN filtering
    """
    def callback(data):
        try:
            # Build performance record
            perf_data = {
                'STRATEGY_NAME': strategy_name,
                'TIMESTAMP': data['timestamp'],
                'PORTFOLIO_VALUE': data['portfolio_value'],
                'CASH': data['cash'],
                'IS_OUT_OF_SAMPLE': data['is_out_of_sample']
            }
            
            # Upload performance
            uploader.upload_performance(pd.DataFrame([perf_data]))
            
            # Upload positions
            positions = data.get('positions', {})
            if positions:
                pos_records = []
                for symbol, pos_info in positions.items():
                    # Skip if any critical value is NaN
                    quantity = pos_info.get('quantity', 0)
                    market_value = pos_info.get('value', 0)
                    avg_price = pos_info.get('avg_price', 0)
                    
                    if not is_valid_number(quantity) or not is_valid_number(market_value) or not is_valid_number(avg_price):
                        print(f"   ⚠️  Skipping position {symbol} due to NaN values")
                        continue
                    
                    pos_records.append({
                        'STRATEGY_NAME': strategy_name,
                        'TIMESTAMP': data['timestamp'],
                        'SYMBOL': symbol,
                        'QUANTITY': quantity,
                        'MARKET_VALUE': market_value,
                        'AVG_PRICE': avg_price
                    })
                
                if pos_records:
                    uploader.upload_positions(pd.DataFrame(pos_records))
            
            # Upload predictions (ML strategy only)
            predictions = data.get('predictions', {})
            if predictions and predictions.get('symbols'):
                pred_records = []
                symbols_list = predictions.get('symbols', [])
                returns_list = predictions.get('returns', [])
                vol_list = predictions.get('volatility', [])
                
                for i, symbol in enumerate(symbols_list):
                    # Skip if symbol is invalid
                    if not symbol or pd.isna(symbol):
                        continue
                    
                    # Get prediction values
                    pred_return = returns_list[i] if i < len(returns_list) else None
                    pred_vol = vol_list[i] if i < len(vol_list) else None
                    
                    # CRITICAL: Skip this prediction if ANY value is NaN
                    # We only include predictions with BOTH valid return AND volatility
                    if not is_valid_number(pred_return) or not is_valid_number(pred_vol):
                        print(f"   ⚠️  Skipping prediction for {symbol} due to NaN (return={pred_return}, vol={pred_vol})")
                        continue
                    
                    pred_records.append({
                        'STRATEGY_NAME': strategy_name,
                        'TIMESTAMP': data['timestamp'],
                        'SYMBOL': symbol,
                        'PREDICTED_RETURN': float(pred_return),
                        'PREDICTED_VOL': float(pred_vol)
                    })
                
                if pred_records:
                    print(f"   📊 Uploading {len(pred_records)} valid predictions (skipped {len(symbols_list) - len(pred_records)} NaN predictions)")
                    uploader.upload_predictions(pd.DataFrame(pred_records))
                else:
                    print(f"   ⚠️  No valid predictions to upload (all {len(symbols_list)} had NaN values)")
            
            print(f"✅ Logged {strategy_name} data for {data['timestamp']}")
            
        except Exception as e:
            print(f"❌ Error logging {strategy_name}: {e}")
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