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

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def _build_features_from_pipeline(self):
        """
        Pull data from your pipeline and construct ML training matrices.
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

    def on_trading_iteration(self):
        if self.optimizer is None:
            print("⚠️ No optimizer set. Skipping iteration.")
            return

        # Avoid retraining every bar
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

        # Train
        # Clean labels to avoid XGBoost NaN/inf crash
        mask = (
            np.isfinite(y_ret.values) &
            np.isfinite(y_vol.values)
        )

        X_clean = X.loc[mask]
        y_ret_clean = y_ret.loc[mask]
        y_vol_clean = y_vol.loc[mask]

        if len(X_clean) < 50:
            self.log(f"⚠️ Not enough clean samples ({len(X_clean)}). Skipping iteration.")
            return

        self.optimizer.fit(X_clean, y_ret_clean, y_vol_clean)

        # Predict
        preds = self.optimizer.predict(X_clean)

        # Convert predictions to weights (IMPORTANT)
        weights = self.optimizer.optimal_weights(preds, symbols)

        budget = self.portfolio_value
        orders_placed = 0

        for sym, w in weights.items():
            if abs(w) < 0.02:
                continue

            price = self.get_last_price(sym)
            if price is None or price <= 0:
                continue

            qty = int((w * budget) / price)
            if qty == 0:
                continue

            side = "buy" if qty > 0 else "sell"
            order = self.create_order(sym, abs(qty), side)
            self.submit_order(order)
            orders_placed += 1

        print(f"✅ ML placed {orders_placed} orders")
