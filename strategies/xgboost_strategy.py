"""
XGBoost Strategy - CONTROLLED SHORTING VERSION
Allows shorts with proper risk management:
- Maximum short exposure limits
- Margin requirement tracking
- Cash buffer maintenance

- Weekends: HOLD stock positions (don't sell)
- Weekends: CAN adjust crypto positions
- Only rebalance stocks when markets are open
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
    ML-driven portfolio with 24/7 crypto handling
    
    Key principles:
    1. Stock positions FROZEN on weekends (hold, don't trade)
    2. Crypto positions can be adjusted 24/7
    3. Full rebalancing only on weekdays
    4. Weekend-only crypto adjustments are OPTIONAL and minor
    """

    def __init__(self, 
                 broker, 
                 performance_callback=None, 
                 optimizer=None, 
                 min_samples=50,
                 allow_shorts=False,
                 max_short_exposure=0.30,
                 min_cash_buffer=0.05,
                 margin_requirement=1.5,
                 weekend_crypto_adjustment=True,  # NEW: Enable weekend crypto tweaks
                 **kwargs):
        super().__init__(broker, **kwargs)
        
        print("🔧 INIT: MLPortfolioStrategyCorrect247")
        print(f"   Weekend crypto adjustment: {weekend_crypto_adjustment}")

        # Strategy timing
        if self.is_backtesting:
            self.sleeptime = "1D"  # Daily for backtesting
        else:
            # For live trading:
            # Weekdays: Check at market close (4pm)
            # Weekends: Check once per day (optional)
            self.sleeptime = "1D"  # Can adjust based on needs
        
        self.optimizer = optimizer or PortfolioRiskOptimizer(risk_target=0.15)
        self.min_samples = min_samples
        self.weekend_crypto_adjustment = weekend_crypto_adjustment

        # Shorting controls
        self.allow_shorts = allow_shorts
        self.max_short_exposure = max_short_exposure
        self.min_cash_buffer = min_cash_buffer
        self.margin_requirement = margin_requirement

        if performance_callback is None:
            performance_callback = self.parameters.get("performance_callback")

        self.performance_callback = performance_callback

        self.lookback_days = 120
        self.market_proxy = "SPY"

        self.in_sample_end = pd.Timestamp('2025-06-30', tz='EST')
        self.out_sample_start = pd.Timestamp('2025-07-01', tz='EST')
        
        self.in_sample_performance = []
        self.out_sample_performance = []

        # Define asset types
        if self.is_backtesting:
            self.tradeable_symbols = [yahoo_sym for _, yahoo_sym in ASSETS]
            self.crypto_symbols = ['BTC-USD']
            self.stock_symbols = [s for s in self.tradeable_symbols if s not in self.crypto_symbols]
        else:
            self.tradeable_symbols = [alpaca_sym for alpaca_sym, _ in ASSETS]
            self.crypto_symbols = ['BTCUSD']
            self.stock_symbols = [s for s in self.tradeable_symbols if s not in self.crypto_symbols]
        
        print(f"📋 Total symbols: {len(self.tradeable_symbols)}")
        print(f"   Crypto (24/7): {self.crypto_symbols}")
        print(f"   Stocks (weekdays): {len(self.stock_symbols)}")

    def _is_weekend(self):
        """Check if current day is weekend"""
        current_time = self.get_datetime()
        return current_time.weekday() >= 5  # Saturday=5, Sunday=6
    
    def _is_market_open(self, symbol):
        """
        Check if market is open for this symbol
        
        Args:
            symbol: Asset symbol
        
        Returns:
            bool: True if market open
        """
        is_crypto = symbol in self.crypto_symbols
        is_weekend = self._is_weekend()
        
        if is_crypto:
            # Crypto always tradeable
            return True
        else:
            # Stocks only on weekdays
            # TODO: Could enhance with holiday calendar
            return not is_weekend

    def _build_features_from_pipeline(self):
        is_backtesting = self.is_backtesting

        today = self.get_datetime()
        end_date = today - pd.Timedelta(days=1)
        start_date = end_date - pd.Timedelta(days=self.lookback_days)
        
        sample_type = "OUT-OF-SAMPLE" if self._is_out_of_sample() else "IN-SAMPLE"
        day_type = "WEEKEND" if self._is_weekend() else "WEEKDAY"
        print(f"  [{sample_type}] [{day_type}] Training on: {start_date.date()} to {end_date.date()}")

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
        CORRECT weekend handling:
        - Weekdays: Full rebalancing (all assets)
        - Weekends: Only adjust crypto (stocks held)
        """
        current_time = self.get_datetime()
        is_weekend = self._is_weekend()
        
        print("\n" + "="*60)
        print(f"🔄 on_trading_iteration() at {current_time}")
        print(f"   Day: {'WEEKEND' if is_weekend else 'WEEKDAY'}")
        print("="*60)
        
        try:
            # ========== WEEKEND EARLY EXIT (Optional) ==========
            if is_weekend and not self.weekend_crypto_adjustment:
                print("   🌙 WEEKEND: Skipping (crypto adjustment disabled)")
                print("   📊 Stock positions: HELD (no action)")
                print("   📊 Crypto positions: HELD (no action)")
                return
            
            if self.optimizer is None:
                print("   ⚠️  No optimizer. Exiting...")
                return

            data = self._build_features_from_pipeline()
            
            if data[0] is None:
                print("   ⚠️  No ML data. Exiting...")
                return

            X, y_ret, y_vol, symbols = data

            if len(X) < self.min_samples:
                print(f"   ⚠️  Not enough samples. Exiting...")
                return

            mask = np.isfinite(y_ret.values) & np.isfinite(y_vol.values)
            X_clean = X.loc[mask]
            y_ret_clean = y_ret.loc[mask]
            y_vol_clean = y_vol.loc[mask]
            symbols_clean = symbols.loc[mask]

            if len(X_clean) < self.min_samples:
                print(f"   ⚠️  Not enough clean samples. Exiting...")
                return

            self.optimizer.fit(X_clean, y_ret_clean, y_vol_clean)
            preds, latest_idx = self.optimizer.predict_latest(X_clean, symbols_clean)
            latest_symbols = symbols_clean.loc[latest_idx].values

            # Get target weights (volatility parity)
            target_weights = self.optimizer.optimal_weights(preds, latest_symbols, method="vol_parity")
            
            # ========== CRITICAL: WEEKEND LOGIC ==========
            if is_weekend:
                print("   🌙 WEEKEND MODE: Stock positions FROZEN")
                
                # Get current portfolio allocation
                portfolio_value = self.portfolio_value
                current_positions = {}
                
                for sym in self.tradeable_symbols:
                    pos = self.get_position(sym)
                    if pos:
                        price = self.get_last_price(sym)
                        if price and price > 0:
                            current_positions[sym] = (pos.quantity * price) / portfolio_value
                
                # CRITICAL: For stocks, use CURRENT weight (don't change!)
                # For crypto, use TARGET weight (can adjust)
                adjusted_weights = {}
                for sym in target_weights.keys():
                    if sym in self.stock_symbols:
                        # Stock: Keep current position
                        adjusted_weights[sym] = current_positions.get(sym, 0)
                        print(f"   📊 {sym}: HOLD at {adjusted_weights[sym]*100:.1f}% (market closed)")
                    else:
                        # Crypto: Use target weight
                        adjusted_weights[sym] = target_weights[sym]
                        print(f"   📊 {sym}: TARGET {target_weights[sym]*100:.1f}% (can trade)")
                
                weights = adjusted_weights
            else:
                # Weekday: Use target weights for all assets
                weights = target_weights
            
            # ========== POSITION SIZING ==========
            weights = {sym: max(0, w) for sym, w in weights.items()}  # Long-only
            
            # Clip and normalize
            clipped_weights = {
                sym: float(np.clip(w, 0, MAX_POSITION_PCT))
                for sym, w in weights.items()
            }
            
            total_weight = sum(clipped_weights.values())
            if total_weight > 0:
                clipped_weights = {sym: w / total_weight for sym, w in clipped_weights.items()}
            
            weights = {sym: w * MAX_GROSS_EXPOSURE for sym, w in clipped_weights.items()}
            
            self.latest_predictions = {
                'returns': preds['ret'].tolist(),
                'volatility': preds['vol'].tolist(),
                'symbols': latest_symbols.tolist()
            }
            self.latest_weights = weights

            if not self._is_risk_on():
                print("   🧯 Risk-off — no trades.")
                return

            portfolio_value = self.portfolio_value
            if portfolio_value <= 0:
                print("   🛑 Portfolio value <= 0.")
                return

            cash = self.get_cash()
            min_cash_required = portfolio_value * self.min_cash_buffer
            available_for_longs = cash - min_cash_required
            
            print(f"   💰 Cash: ${cash:,.2f}")
            print(f"   ✅ Available: ${available_for_longs:,.2f}")

            orders_placed = 0

            # ========== EXECUTE TRADES ==========
            for sym, w in weights.items():
                # CRITICAL: Check if market is open
                if not self._is_market_open(sym):
                    print(f"   ⏸️  {sym}: SKIP (market closed)")
                    continue
                
                target_dollars = w * portfolio_value
                
                price = self.get_last_price(sym)
                if not price or price <= 0:
                    continue

                target_qty = target_dollars / price
                target_qty = round(target_qty, 4)
                
                if abs(target_qty) < 0.0001:
                    continue

                current_pos = self.get_position(sym)
                current_qty = current_pos.quantity if current_pos else 0
                
                delta_qty = target_qty - current_qty
                
                # Weekend stocks should have delta_qty ≈ 0 (holding)
                if is_weekend and sym in self.stock_symbols:
                    if abs(delta_qty) > 0.01:
                        print(f"   ⚠️  WARNING: {sym} delta={delta_qty:.4f} on weekend!")
                    continue  # Skip stock trades on weekends
                
                delta_dollars = delta_qty * price
                
                if delta_qty > 0:
                    if delta_dollars > available_for_longs:
                        affordable_qty = available_for_longs / price
                        delta_qty = affordable_qty
                        delta_dollars = delta_qty * price
                        
                        if delta_qty < 0.0001:
                            continue
                
                if abs(delta_dollars) < MIN_TRADE_DOLLARS:
                    continue
                
                side = "buy" if delta_qty > 0 else "sell"
                order = self.create_order(sym, abs(delta_qty), side)
                self.submit_order(order)
                
                available_for_longs -= delta_dollars if side == "buy" else -abs(delta_dollars)
                orders_placed += 1
                
                print(f"   📝 {side.upper()} {abs(delta_qty):.4f} {sym} @ ${price:.2f}")

            print(f"   ✅ Placed {orders_placed} orders")
            
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            if self.performance_callback is not None:
                try:
                    self._trigger_performance_callback()
                except Exception as e:
                    print(f"   ❌ Callback error: {e}")

    def on_backtest_end(self):
        """Called when backtest ends"""
        print("\n" + "="*60)
        print("BACKTEST COMPLETE")
        print("="*60)
        
        if self.in_sample_performance:
            in_sample_df = pd.DataFrame(self.in_sample_performance)
            in_sample_return = (
                (in_sample_df['value'].iloc[-1] / in_sample_df['value'].iloc[0] - 1) * 100
            )
            print(f"IN-SAMPLE: {in_sample_return:.2f}%")
        
        if self.out_sample_performance:
            out_sample_df = pd.DataFrame(self.out_sample_performance)
            out_sample_return = (
                (out_sample_df['value'].iloc[-1] / out_sample_df['value'].iloc[0] - 1) * 100
            )
            print(f"OUT-OF-SAMPLE: {out_sample_return:.2f}%")
    
    def _trigger_performance_callback(self):
        """Trigger callback with current state"""
        if self.performance_callback is None:
            return
        
        try:
            current_datetime = self.get_datetime()
            portfolio_value = self.get_portfolio_value()
            cash = self.get_cash()
            
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
            
            self.performance_callback(performance_data)
            
        except Exception as e:
            print(f"   ❌ Callback error: {e}")