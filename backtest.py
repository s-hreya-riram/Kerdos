from datetime import datetime
from lumibot.backtesting import YahooDataBacktesting
from lumibot.brokers import Alpaca
from config import ALPACA_CONFIG
#from strategies.strategy import Strategy
from strategies.strategy import MLPortfolioStrategy
#from strategies.example_strategy_1 import example_strategy_1


def run_backtest():
    """
    Run a backtest
    """
    broker = Alpaca(ALPACA_CONFIG)

    # Define the backtest environment
    # You can find more information about the backtest environment here: https://lumibot.lumiwealth.com/backtesting.backtesting_function.html
    backtesting_start = datetime(2025, 3, 1)
    backtesting_end = datetime(2025, 4, 15)
    budget = 10000

    # Instantiate the strategy
    strategy = MLPortfolioStrategy(broker=broker)

    strategy.run_backtest(
        YahooDataBacktesting,
        backtesting_start,
        backtesting_end,
        budget=budget,
    )
    strategy.on_backtest_end()

if __name__ == "__main__":
    run_backtest()