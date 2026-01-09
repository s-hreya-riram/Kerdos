from lumibot.strategies import Strategy


class Strategy(Strategy):
    """
    QF5208 Portfolio Management – Strategy Assignment

    Your task:
      - Design and implement your own trading / portfolio strategy.
      - You will mainly work in `initialize` and `on_trading_iteration`.

    Helpful LumiBot documentation:
      - Lifecycle methods (when each function is called):
        https://lumibot.lumiwealth.com/lifecycle_methods.html
      - Strategy methods (how to get data, send orders, log, etc.):
        https://lumibot.lumiwealth.com/strategy_methods.html
      - Strategy properties (fields such as `self.cash`, `self.portfolio_value`, `self.sleeptime`):
        https://lumibot.lumiwealth.com/strategy_properties.html
      - Entities (what an `Asset`, `Order`, `Position`, etc. look like):
        https://lumibot.lumiwealth.com/entities.html

    High‑level pattern (very common in LumiBot strategies):
      1. In `initialize`:
         - Configure how often the strategy runs (`self.sleeptime`).
         - Choose instruments (tickers) and risk limits (max position size, cash buffer, etc.).
         - Optionally load parameters (e.g. from `self.get_parameters()`).
      2. In `on_trading_iteration`:
        - Read current portfolio (cash, positions).
        - Get recent prices or other data.
        - Compute your trading **signals** or ML/DL **model predictions** and turn them into target weights / target positions.
        - Create orders and submit them.
         - Log what you are doing so that you can debug and write your report.
    """

    def initialize(self):
        """
        Called once before the strategy starts trading.

        Design tips:
          - Choose **frequency** via `self.sleeptime`
              e.g. "1D" (once per trading day), "60M" (once per hour), "5M" (every 5 minutes).
          - Decide **what you can trade**
              e.g. a list of tickers, sectors, or asset classes.
          - Decide basic **risk constraints**
              e.g. max weight per asset, minimum cash buffer, leverage rules.
          - (Optional) Store configuration in attributes so that `on_trading_iteration` is easy to read.

        Useful references:
          - Lifecycle methods:
              https://lumibot.lumiwealth.com/lifecycle_methods.html
          - Strategy properties (what `self.sleeptime`, `self.cash`, etc. mean):
              https://lumibot.lumiwealth.com/strategy_properties.html
        """

        # === EXAMPLE CONFIGURATION PATTERN ===
        #
        # How often should this strategy run?
        # once per trading day?
        # self.sleeptime = "1D"
        #
        # What assets are we allowed to hold?
        # self.target_assets = ["SPY", "QQQ"]
        #
        # Basic risk controls:
        #   - Maximum fraction of portfolio value in any single asset.
        #   - Minimum cash buffer to cover fees / slippage and reduce risk.
        # self.max_weight_per_asset = 0.6     # e.g. ≤ 60% in any one asset
        # self.min_cash_buffer = 0.05         # e.g. keep ≥ 5% in cash
        #
        # You can also pre‑compute or store anything you will re‑use later, for example:
        #   - Lookback windows (e.g. 20‑day moving average).
        #   - Parameter grids (e.g. momentum thresholds).
        #   - State variables (e.g. last rebalance date).
        #
        # Example of a custom state variable:
        # self.lookback_days = 20

        # TODO: replace this with your own initialization logic

        self.log_message("Initialized Strategy")

    def on_trading_iteration(self):
        """
        Main trading loop – called every `self.sleeptime` during market hours.

        Typical structure for this method:
          1. **Observe current state**
             - Read `self.get_portfolio_value()`, `self.get_cash()`.
             - Inspect open positions via `self.get_positions()`.
          2. **Get market data**
             - Use methods such as:
                 - `self.get_last_price("SPY")`
                 - `self.get_historical_prices("SPY", length=..., timestep=...)`
          3. **Compute signals / model predictions / targets**
             - Example ideas:
                 - Moving‑average crossover (trend following).
                 - Mean reversion on recent returns.
                 - Factor scores (value, momentum, quality, etc.).
             - Or use ML/DL models (logistic / linear regression, tree‑based models, neural nets, etc.) that output
               probabilities, expected returns, or scores.
             - Translate your signal or model output into **target position sizes or weights**.
          4. **Turn targets into orders**
             - Compare target positions vs current positions.
             - Use order methods:
                 - `self.create_order(...)`
                 - `self.submit_order(order)` or `self.submit_orders([order1, order2, ...])`
             - Entities docs show what an `Order` or `Position` looks like:
                 https://lumibot.lumiwealth.com/entities.html
          5. **Log what happened**
             - Use `self.log_message(...)` to record decisions, prices, and P&L.
             - Good logging makes it much easier to debug and to write up your report.

        Tip:
          - Start simple (e.g. 1‑2 assets, basic rule), then iterate.
          - Use the example strategies in the `strategies/` folder as concrete templates.
        """

        # === EXAMPLE SCAFFOLD =========
        #
        # 1. Read current portfolio state
        # portfolio_value = self.get_portfolio_value()
        # cash = self.get_cash()
        # positions = self.get_positions()
        #
        # 2. Get latest prices or historical data for your assets
        # price_spy = self.get_last_price("SPY")
        # hist_spy = self.get_historical_prices("SPY", length=self.lookback_days, timestep="1D")
        #
        # 3. Compute your model prediction / signal and target position
        # signal_or_score = ...  # e.g. indicator‑based signal OR ML/DL model output
        # target_weight_spy = ...  # between 0 and 1 (or -1 and 1 if you short)
        #
        # 4. Convert target weights to target quantities and create / submit orders
        # current_position_spy = self.get_position("SPY")
        # current_qty = current_position_spy.quantity if current_position_spy is not None else 0
        # target_qty = ...  # based on target_weight_spy * portfolio_value / price_spy
        #
        # if target_qty != current_qty:
        #     order = self.create_order(
        #         asset="SPY",
        #         quantity=target_qty - current_qty,
        #         side="buy" if target_qty > current_qty else "sell",
        #     )
        #     self.submit_order(order)
        #
        # 5. Log what you did (even if you decided to do nothing)
        # self.log_message(
        #     f"Signal={signal}, price={price_spy}, "
        #     f"target_qty={target_qty}, current_qty={current_qty}, "
        #     f"portfolio_value={portfolio_value}, cash={cash}"
        # )

        # For now, we at least log the current portfolio; this is a safe default
        # that you can keep even after adding your own trading logic.
        self.log_message(
            f"[Strategy] "
            f"Portfolio=${self.get_portfolio_value():,.2f}, Cash=${self.get_cash():,.2f}, "
            f"Positions={self.get_positions()}"
        )
