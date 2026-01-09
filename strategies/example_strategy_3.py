from lumibot.strategies import Strategy


class example_strategy_3(Strategy):
    """
    Example Strategy 3 – MAG7 equal‑weight portfolio.

    Behavior:
      - Once per trading day, rebalance into an equal‑weight portfolio
        of the “Magnificent 7” mega‑cap tech stocks:
        AAPL, MSFT, AMZN, GOOGL, META, NVDA, TSLA.
      - Uses fractional shares so the portfolio tracks an equal‑weight MAG7 index.
    """

    def initialize(self):
        # Run once per trading day
        self.sleeptime = "1D"

        # Magnificent 7 tickers
        self.mag7 = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA"]

    def on_trading_iteration(self):
        portfolio_value = self.get_portfolio_value()
        if portfolio_value <= 0:
            return

        # Equal weight for each MAG7 stock
        n = len(self.mag7)
        target_weight = 1.0 / n

        for symbol in self.mag7:
            price = self.get_last_price(symbol)
            if price is None or price <= 0:
                # Skip this symbol if we don't have valid pricing
                continue

            # Target dollar allocation and target (possibly fractional) quantity
            target_value = portfolio_value * target_weight
            target_qty = target_value / price

            position = self.get_position(symbol)
            current_qty = position.quantity if position is not None else 0.0

            delta_qty = target_qty - current_qty
            # Small tolerance to avoid tiny trades
            if abs(delta_qty) <= 1e-6:
                continue

            side = "buy" if delta_qty > 0 else "sell"
            trade_qty = abs(delta_qty)

            order = self.create_order(symbol, trade_qty, side)
            self.submit_order(order)
