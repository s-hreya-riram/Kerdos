"""
XGBoost Strategy - DEBUG VERSION with extensive logging
This version has print statements everywhere to trace execution
"""

import numpy as np
import pandas as pd
from lumibot.strategies.strategy import Strategy

from data.data_fetcher import fetch_asset_data
from data.data_pipeline import calculate_basic_features, create_cross_asset_features, prepare_ml_features
from data.constants import ASSETS, MAX_GROSS_EXPOSURE, MAX_POSITION_PCT, MIN_TRADE_DOLLARS
from data.model import PortfolioRiskOptimizer

class MLPortfolioStrategy(Strategy):
    """
    ML-driven portfolio strategy - DEBUG VERSION
    Prints extensive logs to trace callback execution
    """

    def __init__(self, broker, performance_callback=None, optimizer=None, min_samples=50, **kwargs):
        super().__init__(broker, **kwargs)
        
        print("🔧 INIT: Creating MLPortfolioStrategy")
        print(f"   performance_callback provided: {performance_callback is not None}")

        self.sleeptime = "1D"
        self.optimizer = optimizer or PortfolioRiskOptimizer(risk_target=0.15)
        self.min_samples = min_samples

        if performance_callback is None:
            performance_callback = self.parameters.get("performance_callback")

        self.performance_callback = performance_callback
        
        print(f"   self.performance_callback set: {self.performance_callback is not None}")

        self.features = None
        self.targets = None

        self.lookback_days = 120

        self.market_proxy = "SPY"

        self.in_sample_end = pd.Timestamp('2025-06-30', tz='EST')
        self.out_sample_start = pd.Timestamp('2025-07-01', tz='EST')
        
        self.in_sample_performance = []
        self.out_sample_performance = []

        if self.is_backtesting:
            self.tradeable_symbols = [yahoo_sym for _, yahoo_sym in ASSETS]
        else:
            self.tradeable_symbols = [alpaca_sym for alpaca_sym, _ in ASSETS]
        
        print(f"📋 Tradeable symbols: {self.tradeable_symbols}")

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def _build_features_from_pipeline(self):
        is_backtesting = self.is_backtesting

        today = self.get_datetime()
        end_date = today - pd.Timedelta(days=1)
        start_date = end_date - pd.Timedelta(days=self.lookback_days)
        
        sample_type = "OUT-OF-SAMPLE" if self._is_out_of_sample() else "IN-SAMPLE"
        print(f"  [{sample_type}] Training on: {start_date.date()} to {end_date.date()}")

        symbol_mapping = ASSETS

        raw_data = fetch_asset_data(
            symbol_mapping=symbol_mapping,
            is_backtesting=is_backtesting,
            start_date=start_date,
            end_date=end_date
        )

        if not raw_data:
            return None, None, None, None

        featured = {}
        for sym, df in raw_data.items():
            featured[sym] = calculate_basic_features(df)

        enhanced = create_cross_asset_features(featured)
        ml_df = prepare_ml_features(enhanced)

        if ml_df.empty:
            return None, None, None, None
        
        latest_data_date = ml_df.index.max()
        if is_backtesting:
            days_gap = (today - latest_data_date).days
            if days_gap < 1:
                raise ValueError(
                    f"⚠️ DATA LEAKAGE! Latest training data is {latest_data_date.date()} "
                    f"but today is {today.date()}. Gap should be >= 1 day."
                )

        X = ml_df.drop(columns=["symbol", "target_return", "target_volatility"])
        y_ret = ml_df["target_return"]
        y_vol = ml_df["target_volatility"]

        return X, y_ret, y_vol, ml_df["symbol"]

    def _is_risk_on(self):
        bars = self.get_historical_prices(self.market_proxy, 250, "day")
        if bars is None or len(bars) < 200:
            return True

        close = bars.df["close"]
        ma200 = close.rolling(200).mean()
        return close.iloc[-1] > ma200.iloc[-1]
    
    def _is_out_of_sample(self):
        now = self.get_datetime()
        return now >= self.out_sample_start

    def on_trading_iteration(self):
        """
        CRITICAL: Callback is in finally block
        """
        print("\n" + "="*60)
        print(f"🔄 on_trading_iteration() CALLED at {self.get_datetime()}")
        print(f"   callback exists: {self.performance_callback is not None}")
        print("="*60)
        
        try:
            print("   📍 ENTERING try block...")
            
            # ========== TRADING LOGIC ==========
            if self.optimizer is None:
                print("   ⚠️  No optimizer set. Exiting early...")
                return

            data = self._build_features_from_pipeline()
            
            if data[0] is None:
                print("   ⚠️  ML pipeline returned no data. Exiting early...")
                return

            X, y_ret, y_vol, symbols = data

            if len(X) < self.min_samples:
                print(f"   ⚠️  Not enough samples ({len(X)}). Exiting early...")
                return

            mask = np.isfinite(y_ret.values) & np.isfinite(y_vol.values)
            X_clean = X.loc[mask]
            y_ret_clean = y_ret.loc[mask]
            y_vol_clean = y_vol.loc[mask]
            symbols_clean = symbols.loc[mask]

            if len(X_clean) < self.min_samples:
                print(f"   ⚠️  Not enough clean samples ({len(X_clean)}). Exiting early...")
                return

            self.optimizer.fit(X_clean, y_ret_clean, y_vol_clean)

            preds, latest_idx = self.optimizer.predict_latest(X_clean, symbols_clean)
            latest_symbols = symbols_clean.loc[latest_idx].values

            weights = self.optimizer.optimal_weights(preds, latest_symbols)

            # Clip individual position sizes
            clipped_weights = {
                sym: float(np.clip(w, -MAX_POSITION_PCT, MAX_POSITION_PCT))
                for sym, w in weights.items()
            }

            # Renormalize gross exposure
            gross_exposure = sum(abs(w) for w in clipped_weights.values())

            if gross_exposure > MAX_GROSS_EXPOSURE and gross_exposure > 0:
                scale = MAX_GROSS_EXPOSURE / gross_exposure
                clipped_weights = {sym: w * scale for sym, w in clipped_weights.items()}

            weights = clipped_weights
            
            self.latest_predictions = {
                'returns': preds['ret'].tolist(),
                'volatility': preds['vol'].tolist(),
                'symbols': latest_symbols.tolist()
            }
            self.latest_weights = weights

            if not self._is_risk_on():
                print("   🧯 Risk-off regime detected — no trades placed.")
                return

            portfolio_value = self.portfolio_value
            if portfolio_value <= 0:
                print("   🛑 Portfolio value <= 0, halting trading.")
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

            print(f"   ✅ XGB placed {orders_placed} rebalancing orders")

            # Track performance
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
        
        except Exception as e:
            print(f"   ❌ EXCEPTION in try block: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            print("   📍 ENTERING finally block...")
            print(f"   📍 self.performance_callback is not None: {self.performance_callback is not None}")
            
            if self.performance_callback is not None:
                print("   📍 ABOUT TO CALL _trigger_performance_callback()...")
                try:
                    self._trigger_performance_callback()
                    print("   ✅ _trigger_performance_callback() COMPLETED")
                except Exception as e:
                    print(f"   ❌ ERROR in _trigger_performance_callback(): {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print("   ⚠️  Callback is None, skipping")
            
            print("   📍 EXITING finally block")

    def on_backtest_end(self):
        """Called when backtest ends"""
        print("\n" + "="*60)
        print("BACKTEST COMPLETE - PERFORMANCE SUMMARY")
        print("="*60)
        
        if self.in_sample_performance:
            in_sample_df = pd.DataFrame(self.in_sample_performance)
            in_sample_return = (
                (in_sample_df['value'].iloc[-1] / in_sample_df['value'].iloc[0] - 1) * 100
            )
            print(f"IN-SAMPLE:")
            print(f"  Total Return: {in_sample_return:.2f}%")
        
        if self.out_sample_performance:
            out_sample_df = pd.DataFrame(self.out_sample_performance)
            out_sample_return = (
                (out_sample_df['value'].iloc[-1] / out_sample_df['value'].iloc[0] - 1) * 100
            )
            print(f"\nOUT-OF-SAMPLE:")
            print(f"  Total Return: {out_sample_return:.2f}%")
        
        print("="*60 + "\n")
    
    def _trigger_performance_callback(self):
        """Trigger the performance callback with current portfolio state"""
        print("      🎯 _trigger_performance_callback() STARTING...")
        
        if self.performance_callback is None:
            print("      ⚠️  callback is None, returning")
            return
        
        try:
            current_datetime = self.get_datetime()
            portfolio_value = self.get_portfolio_value()
            cash = self.get_cash()
            
            print(f"      📊 Portfolio value: ${portfolio_value:,.2f}")
            print(f"      💵 Cash: ${cash:,.2f}")
            
            # Get positions
            positions_dict = {}
            positions = self.get_positions()
            
            if positions:
                for symbol in self.tradeable_symbols:
                    position = self.get_position(symbol)
                    if position and hasattr(position, 'quantity'):
                        qty = position.quantity
                        if abs(qty) > 0.001:
                            price = self.get_last_price(symbol)
                            if price:
                                positions_dict[symbol] = {
                                    'quantity': float(qty),
                                    'value': float(abs(qty * price)),
                                    'avg_price': float(price)
                                }
            
            print(f"      📦 Positions: {len(positions_dict)}")
            
            # Build performance data
            performance_data = {
                'timestamp': current_datetime,
                'portfolio_value': float(portfolio_value),
                'cash': float(cash),
                'positions': positions_dict,
                'is_out_of_sample': self._is_out_of_sample(),
                'weights': getattr(self, 'latest_weights', {}),
                'predictions': getattr(self, 'latest_predictions', {
                    'returns': [],
                    'volatility': [],
                    'symbols': []
                })
            }
            
            print(f"      📞 CALLING callback function...")
            # Call the callback
            self.performance_callback(performance_data)
            print(f"      ✅ Callback returned successfully")
            
        except Exception as e:
            print(f"      ❌ Performance callback error: {e}")
            import traceback
            traceback.print_exc()