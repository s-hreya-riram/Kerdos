"""
XGBoost Strategy — with Direction Classifier + Regime Filter

Supports two modes:
1. Training mode (load_pretrained=False): Retrains models daily on 90-day rolling window
2. Submission mode (load_pretrained=True): Loads pre-trained models from disk (no retraining)

Both modes are provided. Retraining completes is expected to complete in a few minutes (based on 
~1.5 mins execution time on Macbook Air M3 with 16GB RAM for 48 days of training) for the 
full competition window.
"""

from anyio import current_time
import numpy as np
import pandas as pd
from lumibot.strategies.strategy import Strategy

from datetime import datetime
from data.data_fetcher import fetch_asset_data
from data.data_pipeline import calculate_basic_features, create_cross_asset_features, prepare_ml_features
from data.constants import ASSETS, MAX_GROSS_EXPOSURE, MAX_POSITION_PCT, MIN_TRADE_DOLLARS, CODE_SUBMISSION_DATE
from data.model import PortfolioRiskOptimizer, RegimeFilter


class MLPortfolioStrategy(Strategy):
    """
    ML-driven portfolio with 24/7 crypto handling.

    Key principles:
        1. Stock positions FROZEN on weekends
        2. Crypto positions can be adjusted 24/7
        3. Full rebalancing only on weekdays
        4. Weekend crypto adjustments are optional
        5. RegimeFilter  — continuous risk scaling (CALM / CAUTION / FEAR)
        6. Direction gating — weights suppressed for assets with P(up) < threshold
        7. Pre-trained mode — loads models from disk (retraining by default, but can be disabled)
    """

    # ---- CLASS-LEVEL TRACKERS (Survives engine shutdown) ----
    is_tracker = []
    oos_tracker = []
    total_evaluations = 0
    skipped_evaluations = 0
    total_spread_tax = 0.0

    expected_spreads = {
        "SPY": 0.0002, "GLD": 0.0002, "SLV": 0.0002, "SMH": 0.0002,
        "ZAP": 0.0010, "DFEN": 0.0010,
        "BTC-USD": 0.0020, "BTC/USD": 0.0020
    }

    def __init__(self,
                 broker,
                 performance_callback=None,
                 optimizer=None,
                 load_pretrained=True,
                 pretrained_path='models/portfolio_optimizer.pkl',
                 min_samples=50,
                 allow_shorts=False,
                 max_short_exposure=0.30,
                 min_cash_buffer=0.05,
                 margin_requirement=1.5,
                 weekend_crypto_adjustment=True,
                 regime_calm_threshold=0.12,
                 regime_fear_threshold=0.22,
                 direction_gate_threshold=0.0,
                 **kwargs):
        super().__init__(broker, **kwargs)

        print("🔧 INIT: MLPortfolioStrategy (with classifier + RegimeFilter)")
        print(f"   Weekend crypto adjustment : {weekend_crypto_adjustment}")
        print(f"   Regime calm/fear thresholds  : {regime_calm_threshold} / {regime_fear_threshold}")
        print(f"   Direction gate threshold  : {direction_gate_threshold}")

        self.sleeptime = "1D"

        self.load_pretrained = load_pretrained
        self.pretrained_path = pretrained_path

        if load_pretrained:
            print(f"   🔄 Loading pre-trained models from: {pretrained_path}")
            self.optimizer = PortfolioRiskOptimizer.load(pretrained_path)
            print(f"   ✅ Pre-trained models loaded (no retraining during execution)")
        else:
            print(f"   🔧 Training mode: models will be retrained during execution")
            self.optimizer = optimizer or PortfolioRiskOptimizer(
                risk_target=0.15,
                direction_gate_threshold=direction_gate_threshold,
            )

        self.regime_filter = RegimeFilter(
            calm_threshold=regime_calm_threshold,
            fear_threshold=regime_fear_threshold,
        )

        self.min_samples = min_samples
        self.weekend_crypto_adjustment = weekend_crypto_adjustment
        self.allow_shorts = allow_shorts
        self.max_short_exposure = max_short_exposure
        self.min_cash_buffer = min_cash_buffer
        self.margin_requirement = margin_requirement

        # ---- DEFENSIVE THRESHOLDS ----
        self.kill_switch_pct = 0.06
        self.is_killed = False
        self.drift_hurdle = 0.015
        self.slippage_rates = self.parameters.get("slippage_rates", {})

        if performance_callback is None:
            performance_callback = self.parameters.get("performance_callback")
        self.performance_callback = performance_callback

        self.lookback_days = 90
        self.market_proxy = "SPY"

        self.in_sample_end    = pd.Timestamp('2025-06-30', tz='EST')
        self.out_sample_start = pd.Timestamp('2025-07-01', tz='EST')

        self.in_sample_performance  = []
        self.out_sample_performance = []

        if self.is_backtesting:
            self.tradeable_symbols = [yahoo_sym  for _, yahoo_sym  in ASSETS]
            self.crypto_symbols    = ['BTC-USD']
        else:
            self.tradeable_symbols = [alpaca_sym for alpaca_sym, _ in ASSETS]
            self.crypto_symbols    = ['BTC/USD']

        self.stock_symbols = [s for s in self.tradeable_symbols
                              if s not in self.crypto_symbols]

        print(f"📋 Total symbols : {len(self.tradeable_symbols)}")
        print(f"   Crypto (24/7) : {self.crypto_symbols}")
        print(f"   Stocks        : {len(self.stock_symbols)}")


    def _is_weekend(self):
        return self.get_datetime().weekday() >= 5

    def _is_market_open(self, symbol):
        if symbol in self.crypto_symbols:
            return True
        return not self._is_weekend()

    def _is_out_of_sample(self):
        return self.get_datetime() >= self.out_sample_start


    def _get_regime(self) -> dict:
        bars = self.get_historical_prices(self.market_proxy, 60, "day")
        if bars is None or len(bars) < 22:
            print("   ⚠️  Regime: insufficient SPY data — defaulting to CALM")
            return {"regime": "CALM", "realised_vol": 0.15,
                    "allocation_scale": 1.0, "vix_proxy": 0.15}

        spy_close = bars.df["close"]
        return self.regime_filter.compute(spy_close, window=21)


    def _build_features_from_pipeline(self):
        is_backtesting = self.is_backtesting
        today      = self.get_datetime()
        end_date   = today - pd.Timedelta(days=1)
        start_date = end_date - pd.Timedelta(days=self.lookback_days)

        sample_type = "OOS" if self._is_out_of_sample() else "IS"
        day_type    = "WEEKEND" if self._is_weekend() else "WEEKDAY"

        if self.load_pretrained:
            print(f"  [{sample_type}][{day_type}] Feature window: {start_date.date()} → {end_date.date()} (pre-trained mode)")
        else:
            print(f"  [{sample_type}][{day_type}] Training: {start_date.date()} → {end_date.date()}")

        raw_data = fetch_asset_data(
            symbol_mapping=ASSETS,
            is_backtesting=is_backtesting,
            start_date=start_date,
            end_date=end_date,
        )
        if not raw_data:
            return None, None, None, None

        featured = {sym: calculate_basic_features(df) for sym, df in raw_data.items()}
        enhanced = create_cross_asset_features(featured)
        ml_df    = prepare_ml_features(enhanced)

        if ml_df.empty:
            return None, None, None, None

        latest_data_date = ml_df.index.max()
        if is_backtesting:
            days_gap = (today - latest_data_date).days
            if days_gap < 1:
                raise ValueError(
                    f"⚠️ DATA LEAKAGE! Latest training data {latest_data_date.date()} "
                    f"but today is {today.date()}. Gap must be ≥ 1 day."
                )

        X     = ml_df.drop(columns=["symbol", "target_return", "target_volatility"])
        y_ret = ml_df["target_return"]
        y_vol = ml_df["target_volatility"]

        return X, y_ret, y_vol, ml_df["symbol"]


    def on_trading_iteration(self):
        current_time = self.get_datetime()
        cash = self.get_cash()
        print(f"🔹 {current_time} | Cash available: ${cash:,.2f}")
        is_weekend   = self._is_weekend()

        print("\n" + "=" * 60)
        print(f"🔄 {current_time}  |  {'WEEKEND' if is_weekend else 'WEEKDAY'}")
        print("=" * 60)

        # ---- RECORD EQUITY CURVE ----
        portfolio_value = self.portfolio_value
        perf_record = {"timestamp": current_time, "value": portfolio_value}

        if self._is_out_of_sample():
            MLPortfolioStrategy.oos_tracker.append(perf_record)
        else:
            MLPortfolioStrategy.is_tracker.append(perf_record)

        try:
            # ---- 🚨 PORTFOLIO KILL SWITCH ----
            if not hasattr(self, 'initial_capital'):
                self.initial_capital = portfolio_value

            if self.is_killed:
                print("   💀 STRATEGY DEAD: Kill Switch was previously triggered. Holding Cash.")
                return

            if portfolio_value <= self.initial_capital * (1 - self.kill_switch_pct):
                print(f"   🚨 KILL SWITCH TRIGGERED! Equity dropped below {self.kill_switch_pct:.1%}.")
                for sym in self.tradeable_symbols:
                    pos = self.get_position(sym)
                    if pos and pos.quantity > 0:
                        self.submit_order(self.create_order(sym, pos.quantity, "sell"))
                self.is_killed = True
                return

            if is_weekend and not self.weekend_crypto_adjustment:
                print("   🌙 WEEKEND: skipping (crypto adjustment disabled)")
                return

            if self.optimizer is None:
                print("   ⚠️  No optimizer")
                return

            regime_info = self._get_regime()
            regime      = regime_info["regime"]
            scale       = regime_info["allocation_scale"]
            print(f"   📊 Regime: {regime}  |  Proxy: {regime_info['realised_vol']:.1%}  "
                  f"|  Allocation scale: {scale:.0%}")

            if scale == 0.0:
                print("   🧯 Extreme FEAR — holding cash, no trades.")
                return

            data = self._build_features_from_pipeline()
            if data[0] is None:
                print("   ⚠️  No ML data")
                return

            X, y_ret, y_vol, symbols = data

            if len(X) < self.min_samples:
                print(f"   ⚠️  Not enough samples ({len(X)} < {self.min_samples})")
                return

            mask = np.isfinite(y_ret.values) & np.isfinite(y_vol.values)
            X_clean       = X.loc[mask]
            y_ret_clean   = y_ret.loc[mask]
            y_vol_clean   = y_vol.loc[mask]
            symbols_clean = symbols.loc[mask]

            if len(X_clean) < self.min_samples:
                print(f"   ⚠️  Not enough clean samples")
                return

            if self.load_pretrained:
                print("   🔄 Using pre-trained models (no retraining)")
            else:
                print("   🤖 Training models on current data window")
                self.optimizer.fit(X_clean, y_ret_clean, y_vol_clean)

            preds, latest_symbols = self.optimizer.predict_latest(X_clean, symbols_clean)

            for i, sym in enumerate(latest_symbols):
                print(f"   🔮 {sym:8s}  vol={preds['vol'][i]:.4f}  "
                      f"ret={preds['ret'][i]:+.4f}  P(up)={preds['dir'][i]:.2f}")

            target_weights = self.optimizer.optimal_weights(
                preds, latest_symbols,
                method="vol_parity",
            )

            weights = {sym: max(0, w) for sym, w in target_weights.items()}

            clipped_weights = {
                sym: float(np.clip(w, 0, MAX_POSITION_PCT))
                for sym, w in weights.items()
            }
            total_weight = sum(clipped_weights.values())
            if total_weight > 0:
                clipped_weights = {sym: w / total_weight for sym, w in clipped_weights.items()}

            weights = {
                sym: w * MAX_GROSS_EXPOSURE * scale
                for sym, w in clipped_weights.items()
            }

            if is_weekend:
                print("   🌙 WEEKEND MODE: stock positions frozen")
                portfolio_value   = self.portfolio_value
                current_positions = {}
                for sym in self.tradeable_symbols:
                    pos = self.get_position(sym)
                    if pos:
                        if sym == "BTC/USD":
                            price = self.get_last_price(asset="BTC", quote="USD")
                        else:
                            price = self.get_last_price(sym)
                        if price and price > 0:
                            current_positions[sym] = (pos.quantity * price) / portfolio_value

                adjusted_weights = {}
                for sym, w in weights.items():
                    if sym in self.stock_symbols:
                        adjusted_weights[sym] = current_positions.get(sym, 0)
                        print(f"   📊 {sym}: HOLD {adjusted_weights[sym]*100:.1f}%")
                    else:
                        adjusted_weights[sym] = w
                        print(f"   📊 {sym}: TARGET {w*100:.1f}% (crypto)")

                weights = adjusted_weights

            self.latest_predictions = {
                "returns":    preds["ret"].tolist(),
                "volatility": preds["vol"].tolist(),
                "direction":  preds["dir"].tolist(),
                "symbols":    latest_symbols.tolist(),
            }
            self.latest_weights = weights
            self.latest_regime  = regime_info

            portfolio_value = self.portfolio_value
            if portfolio_value <= 0:
                print("   🛑 Portfolio value ≤ 0")
                return

            cash = self.get_cash()
            min_cash_required   = portfolio_value * self.min_cash_buffer
            available_for_longs = cash - min_cash_required

            print(f"   💰 Cash: ${cash:,.2f}  |  Available: ${available_for_longs:,.2f}  "
                  f"|  Deployed target: {MAX_GROSS_EXPOSURE * scale:.0%}")

            orders_placed = 0

            for sym, w in weights.items():
                if not self._is_market_open(sym):
                    print(f"   ⏸️  {sym}: SKIP (market closed)")
                    continue

                target_dollars = w * portfolio_value
                if sym in self.crypto_symbols and not self.is_backtesting:
                    from lumibot.entities import Asset
                    crypto_symbol = sym.split('/')[0]
                    crypto_asset = Asset(symbol=crypto_symbol, asset_type="crypto")
                    usd_asset = Asset(symbol="USD", asset_type="crypto")
                    price = self.get_last_price(crypto_asset, quote=usd_asset)
                else:
                    price = self.get_last_price(sym)

                if sym in self.crypto_symbols and not self.is_backtesting:
                    crypto_symbol = sym.split('/')[0]
                    if crypto_symbol == "BTC" and price < 10000:
                        print(f"   ⚠️  {sym}: Price ${price:,.2f} looks suspiciously low. Skipping.")
                        continue

                if not price or price <= 0:
                    print(f"   ⚠️  {sym}: Invalid price {price}")
                    continue

                print(f"   💵 {sym}: ${price:,.2f}")
                decimals = 8 if sym in self.crypto_symbols else 4
                target_qty = round(target_dollars / price, decimals)

                min_qty = 0.00000001 if sym in self.crypto_symbols else 0.0001
                if abs(target_qty) < min_qty:
                    continue

                current_pos = self.get_position(sym)
                current_qty = current_pos.quantity if current_pos else 0
                delta_qty   = target_qty - current_qty

                # ---- DRIFT HURDLE ----
                MLPortfolioStrategy.total_evaluations += 1
                current_weight = (current_qty * price) / portfolio_value
                drift = abs(w - current_weight)

                if drift < self.drift_hurdle:
                    MLPortfolioStrategy.skipped_evaluations += 1
                    print(f"   🛡️ SKIP {sym}: Drift {drift:.1%} is below the {self.drift_hurdle:.1%} hurdle.")
                    continue

                if is_weekend and sym in self.stock_symbols:
                    if abs(delta_qty) > 0.01:
                        print(f"   ⚠️  {sym} delta={delta_qty:.4f} on weekend!")
                    continue

                delta_dollars = delta_qty * price

                if delta_qty > 0 and delta_dollars > available_for_longs:
                    print(f"   ⚠️  {sym}: Need ${delta_dollars:,.2f} but only ${available_for_longs:,.2f} available. Skipping.")
                    continue

                if abs(delta_dollars) < MIN_TRADE_DOLLARS:
                    continue

                # ---- 0.5% MLO & SPREAD TAX ----
                side = "buy" if delta_qty > 0 else "sell"
                slippage_buffer = 0.005

                if side == "buy":
                    limit_price = round(price * (1 + slippage_buffer), 2)
                else:
                    limit_price = round(price * (1 - slippage_buffer), 2)

                if sym in self.crypto_symbols and not self.is_backtesting:
                    from lumibot.entities import Asset
                    crypto_symbol = sym.split('/')[0]
                    order_asset = Asset(symbol=crypto_symbol, asset_type="crypto")
                    quote_asset = Asset(symbol="USD", asset_type="crypto")
                    order = self.create_order(
                        order_asset, abs(delta_qty), side, quote=quote_asset,
                        type="limit", limit_price=limit_price, time_in_force="ioc"
                    )
                else:
                    order = self.create_order(
                        sym, abs(delta_qty), side,
                        type="limit", limit_price=limit_price, time_in_force="ioc"
                    )

                self.submit_order(order)

                trade_value_dollars = abs(delta_qty) * price
                asset_spread = self.expected_spreads.get(sym, 0.001)
                spread_cost = trade_value_dollars * (asset_spread / 2)
                MLPortfolioStrategy.total_spread_tax += spread_cost

                available_for_longs -= delta_dollars if side == "buy" else -abs(delta_dollars)
                orders_placed += 1
                self._log_indicators(weights, regime_info, preds, latest_symbols)
                print(f"   📝 {side.upper()} {abs(delta_qty):.4f} {sym} @ ${price:.2f}")

            print(f"   ✅ {orders_placed} orders placed  |  Regime: {regime}")

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
        print("\n" + "=" * 60)
        print("BACKTEST COMPLETE: STATISTICAL METRICS")
        print("=" * 60)

        import sys
        sys.stdout.flush()
        if MLPortfolioStrategy.oos_tracker:
            gross_value = MLPortfolioStrategy.oos_tracker[-1]["value"]
        else:
            gross_value = self.portfolio_value

        total_tax = MLPortfolioStrategy.total_spread_tax
        net_value = gross_value - total_tax

        print(f"GROSS TERMINAL VALUE:  ${gross_value:,.2f}")
        print(f"TOTAL SPREAD FRICTION: -${total_tax:,.2f}")
        print(f"NET TERMINAL VALUE:    ${net_value:,.2f}")
        print("-" * 60)

        if self.is_killed:
            print("FINAL STATUS: 💀 KILL SWITCH WAS TRIGGERED DURING RUN")
        else:
            print("FINAL STATUS: 🛡️ SAFE (Never breached 6% drawdown)")
        print("-" * 60)

        if MLPortfolioStrategy.is_tracker:
            df_is = pd.DataFrame(MLPortfolioStrategy.is_tracker)
            ret_is = (df_is["value"].iloc[-1] / df_is["value"].iloc[0] - 1) * 100
            print(f"IN-SAMPLE     : {ret_is:.2f}%")

        if MLPortfolioStrategy.oos_tracker:
            df_oos = pd.DataFrame(MLPortfolioStrategy.oos_tracker)
            ret_oos = (df_oos["value"].iloc[-1] / df_oos["value"].iloc[0] - 1) * 100
            print(f"OUT-OF-SAMPLE : {ret_oos:.2f}%")

            print("\n--- 45-DAY DEPLOYMENT SPRINT METRICS (OOS) ---")
            equity = df_oos["value"]
            window_days = 45
            rolling_returns = (equity / equity.shift(window_days)) - 1
            valid_returns = rolling_returns.dropna()

            if not valid_returns.empty:
                win_rate = (valid_returns > 0).mean()

                def get_max_drawdown(window):
                    roll_max = window.cummax()
                    drawdowns = (window - roll_max) / roll_max
                    return drawdowns.min()

                rolling_drawdowns = equity.rolling(window=window_days).apply(get_max_drawdown)
                valid_drawdowns = rolling_drawdowns.dropna()

                prob_ruin = (valid_drawdowns <= -self.kill_switch_pct).mean()
                p95_dd = valid_drawdowns.quantile(0.05)

                print(f"Rolling 45-Day Win Rate:         {win_rate:.1%}")
                print(f"Probability of Ruin (>6% DD):    {prob_ruin:.1%}")
                print(f"95th Percentile Drawdown:        {p95_dd:.1%}")

        print("\n--- EXECUTION EFFICIENCY ---")
        total = MLPortfolioStrategy.total_evaluations
        skipped = MLPortfolioStrategy.skipped_evaluations

        if total > 0:
            skip_rate = skipped / total
            print(f"Total Trade Evaluations: {total}")
            print(f"Trades Skipped (Hurdle): {skipped}")
            print(f"Drift Hurdle Skip Rate:  {skip_rate:.1%}")

            if skip_rate > 0.90:
                print("   ⚠️ WARNING: Your hurdle is skipping >90% of trades. Consider lowering it.")
        else:
            print("No valid trades evaluated.")
        print("=" * 60)


    def _trigger_performance_callback(self):
        if self.performance_callback is None:
            return

        try:
            positions_dict = {}
            for symbol in self.tradeable_symbols:
                position = self.get_position(symbol)
                if position and hasattr(position, "quantity"):
                    qty = position.quantity
                    if abs(qty) > 0.001:
                        price = self.get_last_price(symbol)
                        if price:
                            positions_dict[symbol] = {
                                "quantity":  float(qty),
                                "value":     float(abs(qty * price)),
                                "avg_price": float(price),
                            }

            performance_data = {
                "timestamp":        self.get_datetime(),
                "portfolio_value":  float(self.get_portfolio_value()),
                "cash":             float(self.get_cash()),
                "positions":        positions_dict,
                "is_out_of_sample": self._is_out_of_sample(),
                "weights":          getattr(self, "latest_weights", {}),
                "regime":           getattr(self, "latest_regime", {}),
                "predictions":      getattr(self, "latest_predictions", {
                    "returns": [], "volatility": [], "direction": [], "symbols": []
                }),
            }

            self.performance_callback(performance_data)

        except Exception as e:
            print(f"   ❌ Callback error: {e}")

    def _log_indicators(self, weights: dict, regime_info: dict, preds: dict,
                        latest_symbols: np.ndarray):
        self.add_line("regime_vol", regime_info["realised_vol"])
        self.add_line("regime_scale", regime_info["allocation_scale"])
        self.add_line("portfolio_deployed", sum(weights.values()))

        if self.optimizer.last_val_metrics:
            m = self.optimizer.last_val_metrics
            self.add_line("model_vol_r2", m.get("vol_r2", 0.0))
            self.add_line("model_dir_acc", m.get("dir_acc", 0.0))