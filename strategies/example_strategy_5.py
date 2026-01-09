from lumibot.strategies import Strategy

# Simple ML example using scikit-learn
import numpy as np
from sklearn.linear_model import LogisticRegression


class example_strategy_5(Strategy):
    """
    Example Strategy 5 – Simple ML-based BTC strategy (logistic regression).

    Behavior:
      - Use recent daily returns of BTC as features.
      - Train a logistic regression classifier to predict whether
        the next day's return will be positive (> 0) or not.
      - Adjust BTC exposure based on the model's probability of "up".
    """

    def initialize(self):
        # Run once per trading day
        self.sleeptime = "1D"

        # Yahoo Finance symbol for Bitcoin in USD
        self.symbol = "BTC-USD"

        # How many days of history to use for training
        self.training_days = 365

        # How many most recent returns to use as features
        self.feature_lags = 5

        # Probability thresholds to set target exposure
        self.risk_on_threshold = 0.60   # strong bullish signal
        self.risk_off_threshold = 0.40  # bearish signal

        # Will hold the trained sklearn model
        self.model = None

    # --------- Helper methods ----------

    def _get_closes(self, bars):
        """Extract close prices from whatever object get_historical_prices returns."""
        try:
            return bars["close"]
        except Exception:
            return bars.df["close"]

    def _train_model(self):
        """Train a logistic regression model once, using past BTC data."""
        if self.model is not None:
            return

        bars = self.get_historical_prices(
            self.symbol,
            length=self.training_days,
            timestep="1D",
        )
        closes = self._get_closes(bars)
        returns = closes.pct_change().dropna().values

        # Need enough data to build a dataset
        if len(returns) <= self.feature_lags + 1:
            return

        X = []
        y = []
        for i in range(self.feature_lags, len(returns) - 1):
            # Feature: last `feature_lags` daily returns
            feat = returns[i - self.feature_lags : i]
            # Label: 1 if next day's return > 0, else 0
            label = 1 if returns[i + 1] > 0 else 0
            X.append(feat)
            y.append(label)

        if not X:
            return

        X = np.array(X)
        y = np.array(y)

        # Need at least two classes to train
        if len(np.unique(y)) < 2:
            return

        model = LogisticRegression()
        model.fit(X, y)
        self.model = model

    # --------- Main trading logic ----------

    def on_trading_iteration(self):
        # Train model the first time we have enough data
        self._train_model()
        if self.model is None:
            return

        # Build today's feature vector using the most recent data
        bars = self.get_historical_prices(
            self.symbol,
            length=self.feature_lags + 2,
            timestep="1D",
        )
        closes = self._get_closes(bars)
        returns = closes.pct_change().dropna().values
        if len(returns) < self.feature_lags:
            return

        recent = returns[-self.feature_lags :]
        X_today = recent.reshape(1, -1)

        # Predict probability that next-day return will be positive
        prob_up = self.model.predict_proba(X_today)[0, 1]

        portfolio_value = self.get_portfolio_value()
        if portfolio_value <= 0:
            return

        last_price = self.get_last_price(self.symbol)
        if last_price is None or last_price <= 0:
            return

        position = self.get_position(self.symbol)
        current_qty = position.quantity if position is not None else 0.0

        # Set target exposure based on probability
        if prob_up >= self.risk_on_threshold:
            target_weight = 1.0   # fully in BTC
        elif prob_up <= self.risk_off_threshold:
            target_weight = 0.0   # in cash
        else:
            target_weight = 0.5   # half in BTC, half in cash

        target_qty = (portfolio_value * target_weight) / last_price
        delta_qty = target_qty - current_qty

        # Small tolerance to avoid tiny trades
        if abs(delta_qty) <= 1e-4:
            return

        side = "buy" if delta_qty > 0 else "sell"
        trade_qty = abs(delta_qty)

        order = self.create_order(self.symbol, trade_qty, side)
        self.submit_order(order)

        # Log for students to inspect
        self.log_message(
            f"[example_strategy_5] prob_up={prob_up:.2f}, "
            f"target_weight={target_weight:.2f}, side={side}, "
            f"trade_qty={trade_qty:.6f}, PV={portfolio_value:,.2f}"
        )

