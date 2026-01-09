from lumibot.strategies import Strategy

class example_strategy_1(Strategy):
    """
    Example Strategy 1 – Simple daily DCA into SPY.

    Behavior:
      - Runs once per trading day.
      - Each iteration, buys 1 share of SPY at market.
    """

    def initialize(self):
        # Run once per day (daily DCA into SPY)
        self.sleeptime = "1D"

    def on_trading_iteration(self):
        symbol = "SPY"
        quantity = 1
        side = "buy"

        order = self.create_order(symbol, quantity, side)
        self.submit_order(order)
