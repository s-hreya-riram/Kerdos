"""
XGBoost Strategy — with Direction Classifier + Regime Filter
"""

import numpy as np
import pandas as pd
from lumibot.strategies.strategy import Strategy

from data.data_fetcher import fetch_asset_data
from data.data_pipeline import calculate_basic_features, create_cross_asset_features, prepare_ml_features
from data.constants import ASSETS, MAX_GROSS_EXPOSURE, MAX_POSITION_PCT, MIN_TRADE_DOLLARS
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
                 weekend_crypto_adjustment=True,
                 # Regime thresholds (annualised SPY realised vol)
                 regime_calm_threshold=0.12,   # below 12% ann vol → full exposure
                 regime_fear_threshold=0.22,   # above 22% ann vol → 30% exposure
                 # Direction gate
                 direction_gate_threshold=0.0,
                 **kwargs):
        super().__init__(broker, **kwargs)

        print("🔧 INIT: MLPortfolioStrategy (with classifier + RegimeFilter)")
        print(f"   Weekend crypto adjustment : {weekend_crypto_adjustment}")
        print(f"   Regime calm/fear thresholds  : {regime_calm_threshold} / {regime_fear_threshold}")
        print(f"   Direction gate threshold  : {direction_gate_threshold}")

        self.sleeptime = "1D"

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

        if performance_callback is None:
            performance_callback = self.parameters.get("performance_callback")
        self.performance_callback = performance_callback

        self.lookback_days = 90
        self.market_proxy = "SPY"

        self.in_sample_end    = pd.Timestamp('2025-06-30', tz='EST')
        self.out_sample_start = pd.Timestamp('2025-07-01', tz='EST')

        self.in_sample_performance  = []
        self.out_sample_performance = []

        # Symbol lists
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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_weekend(self):
        return self.get_datetime().weekday() >= 5

    def _is_market_open(self, symbol):
        if symbol in self.crypto_symbols:
            return True
        return not self._is_weekend()

    def _is_out_of_sample(self):
        return self.get_datetime() >= self.out_sample_start

    # ------------------------------------------------------------------
    # Regime  (replaces the old binary _is_risk_on)
    # ------------------------------------------------------------------

    def _get_regime(self) -> dict:
        """
        Fetch SPY prices and run RegimeFilter.
        Falls back to CALM (scale=1.0) if data unavailable.
        """
        bars = self.get_historical_prices(self.market_proxy, 60, "day")
        if bars is None or len(bars) < 22:
            print("   ⚠️  Regime: insufficient SPY data — defaulting to CALM")
            return {"regime": "CALM", "realised_vol": 0.15,
                    "allocation_scale": 1.0, "vix_proxy": 0.15}

        spy_close = bars.df["close"]
        return self.regime_filter.compute(spy_close, window=21)

    # ------------------------------------------------------------------
    # Feature pipeline
    # ------------------------------------------------------------------

    def _build_features_from_pipeline(self):
        is_backtesting = self.is_backtesting
        today      = self.get_datetime()
        end_date   = today - pd.Timedelta(days=1)
        start_date = end_date - pd.Timedelta(days=self.lookback_days)

        sample_type = "OOS" if self._is_out_of_sample() else "IS"
        day_type    = "WEEKEND" if self._is_weekend() else "WEEKDAY"
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

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def on_trading_iteration(self):
        current_time = self.get_datetime()
        is_weekend   = self._is_weekend()

        print("\n" + "=" * 60)
        print(f"🔄 {current_time}  |  {'WEEKEND' if is_weekend else 'WEEKDAY'}")
        print("=" * 60)

        try:
            # ---- Weekend early exit ----
            if is_weekend and not self.weekend_crypto_adjustment:
                print("   🌙 WEEKEND: skipping (crypto adjustment disabled)")
                return

            if self.optimizer is None:
                print("   ⚠️  No optimizer")
                return

            # ---- Regime check ----
            regime_info = self._get_regime()
            regime      = regime_info["regime"]
            scale       = regime_info["allocation_scale"]
            print(f"   📊 Regime: {regime}  |  Proxy: {regime_info['realised_vol']:.1%}  "
                  f"|  Allocation scale: {scale:.0%}")

            if scale == 0.0:
                print("   🧯 Extreme FEAR — holding cash, no trades.")
                return

            # ---- Feature pipeline ----
            data = self._build_features_from_pipeline()
            if data[0] is None:
                print("   ⚠️  No ML data")
                return

            X, y_ret, y_vol, symbols = data

            if len(X) < self.min_samples:
                print(f"   ⚠️  Not enough samples ({len(X)} < {self.min_samples})")
                return

            # ---- Clean ----
            mask = np.isfinite(y_ret.values) & np.isfinite(y_vol.values)
            X_clean       = X.loc[mask]
            y_ret_clean   = y_ret.loc[mask]
            y_vol_clean   = y_vol.loc[mask]
            symbols_clean = symbols.loc[mask]

            if len(X_clean) < self.min_samples:
                print(f"   ⚠️  Not enough clean samples")
                return

            # ---- Fit all three models ----
            self.optimizer.fit(X_clean, y_ret_clean, y_vol_clean)

            # ---- Predict latest per symbol ----
            # predict_latest returns symbol names directly — no index round-trip
            preds, latest_symbols = self.optimizer.predict_latest(X_clean, symbols_clean)

            for i, sym in enumerate(latest_symbols):
                print(f"   🔮 {sym:8s}  vol={preds['vol'][i]:.4f}  "
                      f"ret={preds['ret'][i]:+.4f}  P(up)={preds['dir'][i]:.2f}")

            # ---- Target weights ----
            # optimal_weights() returns weights summing to 1.0.
            # regime_scale applied once below — NOT inside optimal_weights.
            target_weights = self.optimizer.optimal_weights(
                preds, latest_symbols,
                method="vol_parity",
            )

            # ---- Weekend: freeze stock weights ----
            if is_weekend:
                print("   🌙 WEEKEND MODE: stock positions frozen")
                portfolio_value   = self.portfolio_value
                current_positions = {}
                for sym in self.tradeable_symbols:
                    pos = self.get_position(sym)
                    if pos:
                        price = self.get_last_price(sym)
                        if price and price > 0:
                            current_positions[sym] = (pos.quantity * price) / portfolio_value

                adjusted_weights = {}
                for sym, w in target_weights.items():
                    if sym in self.stock_symbols:
                        adjusted_weights[sym] = current_positions.get(sym, 0)
                        print(f"   📊 {sym}: HOLD {adjusted_weights[sym]*100:.1f}%")
                    else:
                        adjusted_weights[sym] = w
                        print(f"   📊 {sym}: TARGET {w*100:.1f}% (crypto)")

                weights = adjusted_weights
            else:
                weights = target_weights

            # ---- Position sizing ----
            # Clip per-asset, renormalise, then apply gross exposure * regime scale.
            # regime_scale applied exactly once here — not inside optimal_weights().
            weights = {sym: max(0, w) for sym, w in weights.items()}

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

            # ---- Store for callback ----
            self.latest_predictions = {
                "returns":    preds["ret"].tolist(),
                "volatility": preds["vol"].tolist(),
                "direction":  preds["dir"].tolist(),
                "symbols":    latest_symbols.tolist(),
            }
            self.latest_weights = weights
            self.latest_regime  = regime_info

            # ---- Portfolio checks ----
            portfolio_value = self.portfolio_value
            if portfolio_value <= 0:
                print("   🛑 Portfolio value ≤ 0")
                return

            cash = self.get_cash()
            min_cash_required   = portfolio_value * self.min_cash_buffer
            available_for_longs = cash - min_cash_required

            print(f"   💰 Cash: ${cash:,.2f}  |  Available: ${available_for_longs:,.2f}  "
                  f"|  Deployed target: {MAX_GROSS_EXPOSURE * scale:.0%}")

            # ---- Execute trades ----
            orders_placed = 0

            for sym, w in weights.items():
                if not self._is_market_open(sym):
                    print(f"   ⏸️  {sym}: SKIP (market closed)")
                    continue

                target_dollars = w * portfolio_value
                price = self.get_last_price(sym)
                if not price or price <= 0:
                    continue

                target_qty  = round(target_dollars / price, 4)
                if abs(target_qty) < 0.0001:
                    continue

                current_pos = self.get_position(sym)
                current_qty = current_pos.quantity if current_pos else 0
                delta_qty   = target_qty - current_qty

                # Weekend: stocks should have no delta — skip
                if is_weekend and sym in self.stock_symbols:
                    if abs(delta_qty) > 0.01:
                        print(f"   ⚠️  {sym} delta={delta_qty:.4f} on weekend!")
                    continue

                delta_dollars = delta_qty * price

                if delta_qty > 0 and delta_dollars > available_for_longs:
                    delta_qty     = available_for_longs / price
                    delta_dollars = delta_qty * price
                    if delta_qty < 0.0001:
                        continue

                if abs(delta_dollars) < MIN_TRADE_DOLLARS:
                    continue

                side  = "buy" if delta_qty > 0 else "sell"
                order = self.create_order(sym, abs(delta_qty), side)
                self.submit_order(order)

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

    # ------------------------------------------------------------------
    # Backtest end
    # ------------------------------------------------------------------

    def on_backtest_end(self):
        print("\n" + "=" * 60)
        print("BACKTEST COMPLETE")
        print("=" * 60)

        if self.in_sample_performance:
            df = pd.DataFrame(self.in_sample_performance)
            ret = (df["value"].iloc[-1] / df["value"].iloc[0] - 1) * 100
            print(f"IN-SAMPLE     : {ret:.2f}%")

        if self.out_sample_performance:
            df = pd.DataFrame(self.out_sample_performance)
            ret = (df["value"].iloc[-1] / df["value"].iloc[0] - 1) * 100
            print(f"OUT-OF-SAMPLE : {ret:.2f}%")

    # ------------------------------------------------------------------
    # Performance callback
    # ------------------------------------------------------------------

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
        # Regime lines — these go in a separate subplot (no asset= arg)
        self.add_line("regime_vol",   regime_info["realised_vol"])
        self.add_line("regime_scale", regime_info["allocation_scale"])
        self.add_line("portfolio_deployed", sum(weights.values()))

        # Per-asset weight and direction probability lines
        for i, sym in enumerate(latest_symbols):
            clean = sym.replace("-", "_").replace("/", "_")
            self.add_line(f"weight_{clean}",   weights.get(sym, 0.0))
            self.add_line(f"dir_prob_{clean}", float(preds["dir"][i]))

        # Model quality lines
        if self.optimizer.last_val_metrics:
            m = self.optimizer.last_val_metrics
            self.add_line("model_vol_r2",  m.get("vol_r2",  0.0))
            self.add_line("model_dir_acc", m.get("dir_acc", 0.0))