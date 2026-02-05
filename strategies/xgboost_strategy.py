import numpy as np
import pandas as pd
from lumibot.strategies.strategy import Strategy

from data.data_fetcher import fetch_asset_data
from data.data_pipeline import calculate_basic_features, create_cross_asset_features, prepare_ml_features
from data.constants import ASSETS
from data.model import PortfolioRiskOptimizer

class MLPortfolioStrategy(Strategy):
    """
    ML-driven portfolio strategy for Lumibot.
    Compatible with backtesting and live trading.
    """

    def __init__(self, broker, optimizer=None, min_samples=50, **kwargs):
        super().__init__(broker, **kwargs)

        self.optimizer = optimizer or PortfolioRiskOptimizer(risk_target=0.15)
        self.min_samples = min_samples

        self.features = None
        self.targets = None

        self.lookback_days = 120
        self.last_train_date = None

        self.market_proxy = "SPY"

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def _build_features_from_pipeline(self):
        """
        Pull data from the data pipeline and construct training matrices.
        """

        is_backtesting = self.is_backtesting
        end_date = self.get_datetime()
        start_date = end_date - pd.Timedelta(days=self.lookback_days)

        symbol_mapping = ASSETS

        raw_data = fetch_asset_data(
            symbol_mapping=symbol_mapping,
            is_backtesting=is_backtesting,
            start_date=start_date,
            end_date=end_date
        )

        if not raw_data:
            return None, None

        featured = {}
        for sym, df in raw_data.items():
            featured[sym] = calculate_basic_features(df)

        enhanced = create_cross_asset_features(featured)
        ml_df = prepare_ml_features(enhanced)

        if ml_df.empty:
            return None, None

        X = ml_df.drop(columns=["symbol", "target_return", "target_volatility"])
        y_ret = ml_df["target_return"]
        y_vol = ml_df["target_volatility"]

        return X, y_ret, y_vol, ml_df["symbol"]

    def _is_risk_on(self):
        bars = self.get_historical_prices(self.market_proxy, 250, "day")
        if bars is None or len(bars) < 200:
            return True  # fail open

        close = bars.df["close"]
        ma200 = close.rolling(200).mean()
        # We want to trade when we have a long-term uptrend
        return close.iloc[-1] > ma200.iloc[-1]

    def on_trading_iteration(self):
        if self.optimizer is None:
            print("⚠️ No optimizer set. Skipping iteration.")
            return

        today = self.get_datetime().date()
        if self.last_train_date == today:
            return
        self.last_train_date = today

        data = self._build_features_from_pipeline()
        if data is None:
            print("⚠️ ML pipeline returned no data.")
            return

        X, y_ret, y_vol, symbols = data

        if len(X) < self.min_samples:
            print(f"⚠️ Not enough samples ({len(X)}).")
            return

        mask = np.isfinite(y_ret.values) & np.isfinite(y_vol.values)
        X_clean = X.loc[mask]
        y_ret_clean = y_ret.loc[mask]
        y_vol_clean = y_vol.loc[mask]
        symbols_clean = symbols.loc[mask]

        if len(X_clean) < self.min_samples:
            print(f"⚠️ Not enough clean samples ({len(X_clean)}).")
            return
        
        print(f"ML data shapes: X={None if X_clean is None else X_clean.shape}")

        self.optimizer.fit(X_clean, y_ret_clean, y_vol_clean)

        # Predict only latest row per symbol
        preds, latest_idx = self.optimizer.predict_latest(X_clean, symbols_clean)
        latest_symbols = symbols_clean.loc[latest_idx].values

        weights = self.optimizer.optimal_weights(preds, latest_symbols)

        print(f"Raw weights: {weights}")

        print(f"Risk on? {self._is_risk_on()} at {self.get_datetime()}")
        # Optional regime filter
        if not self._is_risk_on():
            print("🧯 Risk-off regime detected — no trades placed.")
            return

        # Risk control
        MAX_GROSS_EXPOSURE = 0.95   # never use more than 95% of portfolio
        MAX_POSITION_PCT = 0.30    # never allocate more than 30% to one asset
        MIN_TRADE_DOLLARS = 1000  # avoid dust trades

        portfolio_value = self.portfolio_value
        if portfolio_value <= 0:
            print("🛑 Portfolio value <= 0, halting trading.")
            return

        orders_placed = 0

        for sym, w in weights.items():
            w = float(np.clip(w, -MAX_POSITION_PCT, MAX_POSITION_PCT))

            target_dollars = w * portfolio_value * MAX_GROSS_EXPOSURE

            price = self.get_last_price(sym)
            if not price or price <= 0:
                continue

            target_qty = int(target_dollars / price)
            if target_qty == 0:
                continue

            current_pos = self.get_position(sym)
            current_qty = current_pos.quantity if current_pos else 0

            delta_qty = target_qty - current_qty

            if abs(delta_qty * price) < MIN_TRADE_DOLLARS:
                continue

            side = "buy" if delta_qty > 0 else "sell"
            order = self.create_order(sym, abs(delta_qty), side)
            print(f"Placing order {side} {sym} {abs(delta_qty)}")
            self.submit_order(order)
            orders_placed += 1

        print(f"✅ XGB placed {orders_placed} rebalancing orders")