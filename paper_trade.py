from lumibot.brokers import Alpaca
from lumibot.traders import Trader
from config import ALPACA_CONFIG
from strategies.strategy import Strategy

def run_paper_trading():
    """Run Strategy on a paper Alpaca account."""
    broker = Alpaca(ALPACA_CONFIG)
    strategy = Strategy(broker=broker)

    trader = Trader()
    trader.add_strategy(strategy)
    trader.run_all()

if __name__ == "__main__":
    run_paper_trading()


