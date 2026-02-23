"""
LIVE RUNNER: MLPortfolioStrategy
Connects to Alpaca Paper Trading to verify execution logic.
"""

from lumibot.brokers import Alpaca
from lumibot.traders import Trader
from strategies.strategy import MLPortfolioStrategy
from config import ALPACA_CONFIG


def run_live():
    # 1. Initialize the Broker in Paper Trading mode
    # Ensure your config.py has 'PAPER': True and the correct Paper Keys
    print("Connecting to Alpaca Paper Trading...")
    broker = Alpaca(ALPACA_CONFIG)

    # 2. Instantiate the Strategy
    # We use load_pretrained=True to verify the model execution
    strategy = MLPortfolioStrategy(
        broker=broker,
        load_pretrained=True,
        pretrained_path='models/portfolio_optimizer.pkl',
        weekend_crypto_adjustment=True
    )

    # 3. Initialize the Trader and register the strategy
    trader = Trader()
    trader.add_strategy(strategy)

    print("\n" + "=" * 60)
    print("🚀 STARTING PAPER TRADING ENGINE")
    print("   Target: Verify Crypto MLOs & Sell-First Logic")
    print("=" * 60 + "\n")

    # 4. Run the live loop
    trader.run_all()


if __name__ == "__main__":
    run_live()