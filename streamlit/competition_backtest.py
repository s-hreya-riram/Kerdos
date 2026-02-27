"""
Competition Backtest Runner
Logs to a separate COMPETITION_PERFORMANCE table for clean Feb 28 start
"""

import os
import sys
from datetime import datetime
from lumibot.backtesting import YahooDataBacktesting
from lumibot.brokers import Alpaca
import pandas as pd

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


def create_competition_callback(uploader, strategy_name):
    """
    Callback for competition mode - logs to COMPETITION_PERFORMANCE table
    with $10,000 starting capital normalization
    """
    
    def callback(data):
        timestamp = data['timestamp']
        portfolio_value = data['portfolio_value']
        cash = data['cash']
        positions = data['positions']
        
        current_date = pd.to_datetime(timestamp)
        current_date_utc = to_utc(current_date)

        if current_date_utc.hour != 16 or current_date_utc.minute != 0:
            return
        
        print(f"      💾 Competition: {current_date_utc.date()} | ${portfolio_value:,.2f}")
        
        # Performance data
        perf_data = {
            'STRATEGY_NAME': strategy_name,
            'TIMESTAMP': current_date_utc,
            'PORTFOLIO_VALUE': float(portfolio_value),
            'CASH': float(cash),
        }
        
        perf_df = pd.DataFrame([perf_data])
        
        # Positions data
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
        
        # Upload to competition tables
        try:
            uploader.upload_competition_performance(perf_df)
            if not pos_df.empty:
                uploader.upload_competition_positions(pos_df)
            print(f"         ✅ Uploaded to competition tables")
        except Exception as e:
            print(f"         ❌ Upload error: {e}")
    
    return callback


def run_competition_backtest():
    """
    Run backtest for competition window only (Feb 28 - Apr 17, 2026)
    Logs to separate tables for clean $10k starting point
    """
    
    print("="*60)
    print("COMPETITION BACKTEST")
    print("="*60)
    print(f"Start time: {datetime.now()}")
    
    # Competition parameters
    START_DATE = datetime(2026, 2, 28)
    END_DATE = datetime(2026, 4, 17)
    INITIAL_CAPITAL = 10000
    
    print(f"\n🏆 Competition Window")
    print(f"   Start: {START_DATE.date()}")
    print(f"   End:   {END_DATE.date()}")
    print(f"   Duration: 49 days")
    print(f"   Starting Capital: ${INITIAL_CAPITAL:,}")
    
    # Snowflake uploader
    print("\n🔗 Connecting to Snowflake...")
    uploader = SnowflakeUploader()
    print("✅ Connected")
    
    broker = Alpaca(ALPACA_CONFIG)
    
    # ========== RUN ML STRATEGY ==========
    print("\n" + "="*60)
    print("🎯 RUNNING KERDOS FUND (Competition Mode)")
    print("="*60)
    
    ml_callback = create_competition_callback(uploader, "ML_XGBOOST")
    
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
        print("✅ Kerdos Fund backtest complete")
        
    except Exception as e:
        print(f"❌ Kerdos Fund failed: {e}")
        import traceback
        traceback.print_exc()
    
    # ========== RUN SPY BENCHMARK ==========
    print("\n" + "="*60)
    print("📊 RUNNING SPY BENCHMARK (Competition Mode)")
    print("="*60)
    
    spy_callback = create_competition_callback(uploader, "SPY_BENCHMARK")
    
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
    print("✅ COMPETITION BACKTEST COMPLETE")
    print("="*60)
    print(f"End time: {datetime.now()}")
    print("\n📊 Data uploaded to:")
    print("   - COMPETITION_PERFORMANCE")
    print("   - COMPETITION_POSITIONS")
    print("\n🎯 Next Steps:")
    print("   1. Update Streamlit to read from competition tables")
    print("   2. Show Feb 28 as day 0 with $10,000 starting value")
    print("   3. Display progress toward Apr 17 end date")


if __name__ == "__main__":
    run_competition_backtest()