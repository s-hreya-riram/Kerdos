from lumibot.strategies import Strategy


class example_strategy_4(Strategy):
    """
    Example Strategy 4 – Permanent / All‑Weather style portfolio.

    Idea (inspired by Harry Browne's “Permanent Portfolio” concept):
      - Hold a diversified, static allocation across:
          * Stocks  (SPY)
          * Long-term bonds (TLT)
          * Gold (GLD)
          * Short-term bonds / cash-like (SHY)
      - Rebalance periodically back to target weights.

    Behavior:
      - Once per trading day, rebalance towards these target weights:
          SPY: 25%, TLT: 25%, GLD: 25%, SHY: 25%
      - Uses fractional shares so it closely tracks the theoretical allocation.
    """

    def initialize(self):
        # Run once per trading day (you could also choose weekly/monthly)
        self.sleeptime = "1D"

        # Target weights for a simple “permanent portfolio” style allocation
        self.target_weights = {
            "SPY": 0.25,  # stocks
            "TLT": 0.25,  # long-term bonds
            "GLD": 0.25,  # gold
            "SHY": 0.25,  # short-term bonds / cash-like
        }

        # Optional: threshold to avoid trading on very small drifts
        self.min_weight_diff = 0.01  # 1% drift tolerance

    def on_trading_iteration(self):
        portfolio_value = self.get_portfolio_value()
        if portfolio_value <= 0:
            return

        for symbol, target_weight in self.target_weights.items():
            price = self.get_last_price(symbol)
            if price is None or price <= 0:
                # Skip if we don't have a valid price
                continue

            # Target dollar allocation and target (fractional) quantity
            target_value = portfolio_value * target_weight
            target_qty = target_value / price

            position = self.get_position(symbol)
            current_qty = position.quantity if position is not None else 0.0
            current_value = current_qty * price
            current_weight = current_value / portfolio_value if portfolio_value > 0 else 0.0

            # If we're already close to target weight, skip to reduce churn
            if abs(current_weight - target_weight) < self.min_weight_diff:
                continue

            delta_qty = target_qty - current_qty
            if abs(delta_qty) <= 1e-6:
                continue

            side = "buy" if delta_qty > 0 else "sell"
            trade_qty = abs(delta_qty)

            order = self.create_order(symbol, trade_qty, side)
            self.submit_order(order)


