from lumibot.strategies import Strategy


class example_strategy_2(Strategy):
    """
    Example Strategy 2 – Simple buy‑and‑hold into SPY.

    Behavior:
      - On the first iteration, invest (almost) all available cash into SPY.
      - After that, do nothing (classic buy & hold).
    """

    def initialize(self):
        # Run once per trading day (but we only trade on the first iteration)
        self.sleeptime = "1D"

        # Track whether we've already done the initial buy
        self.has_bought = False

    def on_trading_iteration(self):
        # If we've already gone all‑in once, do nothing
        if self.has_bought:
            return

        cash = self.get_cash()
        if cash <= 0:
            return

        price = self.get_last_price("SPY")
        if price is None or price <= 0:
            return

        # Use fractional shares so we invest (almost) exactly all cash in SPY
        quantity = cash / price
        if quantity <= 0:
            return

        order = self.create_order("SPY", quantity, "buy")
        self.submit_order(order)

        # Mark that we've completed the one‑time buy
        self.has_bought = True

        self.log_message(
            f"[example_strategy_2] Executed initial buy‑and‑hold in SPY. "
            f"Cash={self.get_cash():,.2f}, Positions={self.get_positions()}"
        )
