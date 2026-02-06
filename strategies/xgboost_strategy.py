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

        # TODO revisit these dates as more data becomes available
        self.in_sample_end = pd.Timestamp('2024-12-31', tz='EST')
        self.out_sample_start = pd.Timestamp('2025-01-01', tz='EST')
        
        # Track performance separately
        self.in_sample_performance = []
        self.out_sample_performance = []


    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def _build_features_from_pipeline(self):
        """
        Pull data from the data pipeline and construct training matrices.
        """

        is_backtesting = self.is_backtesting

        # To prevent lookahead bias, we train on data up to the day before the current date
        today = self.get_datetime()
        end_date = today - pd.Timedelta(days=1)
        start_date = end_date - pd.Timedelta(days=self.lookback_days)
        
        # Log what period we're in
        sample_type = "OUT-OF-SAMPLE" if self._is_out_of_sample() else "IN-SAMPLE"
        print(f"  [{sample_type}] Training on: {start_date.date()} to {end_date.date()}")
        print(f"  Predicting for: {today.date()}")

        symbol_mapping = ASSETS

        raw_data = fetch_asset_data(
            symbol_mapping=symbol_mapping, # since we're using Yahoo for backtesting, Alpaca for paper trading
            is_backtesting=is_backtesting, # to identify the symbols to use
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
        
        latest_data_date = ml_df.index.max()
        # To ensure no data leakage occurs during backtesting
        if is_backtesting:
            days_gap = (today - latest_data_date).days
            if days_gap < 1:
                raise ValueError(
                    f"⚠️ DATA LEAKAGE! Latest training data is {latest_data_date.date()} "
                    f"but today is {today.date()}. Gap should be >= 1 day."
                )
            print(f"  ✅ No leakage: {days_gap} day gap between training data and prediction date")

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
    
    def _is_out_of_sample(self):
        now = self.get_datetime()
        return now >= self.out_sample_start

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

        # narrowing to rows without NaNs
        mask = np.isfinite(y_ret.values) & np.isfinite(y_vol.values)
        X_clean = X.loc[mask]
        y_ret_clean = y_ret.loc[mask]
        y_vol_clean = y_vol.loc[mask]
        symbols_clean = symbols.loc[mask]

        if len(X_clean) < self.min_samples:
            print(f"⚠️ Not enough clean samples ({len(X_clean)}).")
            return

        self.optimizer.fit(X_clean, y_ret_clean, y_vol_clean)

        # Predict the returns and volatility only for the latest row per symbol
        preds, latest_idx = self.optimizer.predict_latest(X_clean, symbols_clean)
        latest_symbols = symbols_clean.loc[latest_idx].values

        weights = self.optimizer.optimal_weights(preds, latest_symbols)

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
            self.submit_order(order)
            orders_placed += 1

        print(f"✅ XGB placed {orders_placed} rebalancing orders")

        if self._is_out_of_sample():
            self.out_sample_performance.append({
                'date': self.get_datetime(),
                'value': portfolio_value
            })
        else:
            self.in_sample_performance.append({
                'date': self.get_datetime(),
                'value': portfolio_value
            })

    # Summarize performance at the end of backtest
    def on_backtest_end(self):
        """Called when backtest ends - print performance summary"""
        print("\n" + "="*60)
        print("BACKTEST COMPLETE - PERFORMANCE SUMMARY")
        print("="*60)
        
        if self.in_sample_performance:
            in_sample_df = pd.DataFrame(self.in_sample_performance)
            in_sample_return = (
                (in_sample_df['value'].iloc[-1] / in_sample_df['value'].iloc[0] - 1) * 100
            )
            print(f"IN-SAMPLE (2020-2024):")
            print(f"  Total Return: {in_sample_return:.2f}%")
        
        if self.out_sample_performance:
            out_sample_df = pd.DataFrame(self.out_sample_performance)
            out_sample_return = (
                (out_sample_df['value'].iloc[-1] / out_sample_df['value'].iloc[0] - 1) * 100
            )
            print(f"\nOUT-OF-SAMPLE (2025):")
            print(f"  Total Return: {out_sample_return:.2f}%")
        
        print("="*60 + "\n")