"""
XGBoost Strategy — with Direction Classifier + Regime Filter

Final Production Build:
1. Two-Phase Execution: Sells all overweights first to maximize available cash.
2. Partial Fill Logic: Scales down buy orders to fit remaining budget instead of skipping.
3. Holiday Awareness: Uses broker.is_market_open to respect exchange calendars.
4. Tight Rebalancing: 0.1% Drift Hurdle optimized for daily-retraining models.
"""

import numpy as np
import pandas as pd
from lumibot.strategies.strategy import Strategy
from datetime import datetime
from data.data_fetcher import fetch_asset_data
from data.data_pipeline import calculate_basic_features, create_cross_asset_features, prepare_ml_features
from data.constants import ASSETS, MAX_GROSS_EXPOSURE, MAX_POSITION_PCT, MIN_TRADE_DOLLARS
from data.model import PortfolioRiskOptimizer, RegimeFilter


class MLPortfolioStrategy(Strategy):
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
                 load_pretrained=False,
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

        print("🔧 INIT: MLPortfolioStrategy (Optimized Execution Build)")

        self.sleeptime = "1D"
        self.load_pretrained = load_pretrained
        self.pretrained_path = pretrained_path

        if load_pretrained:
            self.optimizer = PortfolioRiskOptimizer.load(pretrained_path)
        else:
            self.optimizer = optimizer or PortfolioRiskOptimizer(
                risk_target=0.15,
                direction_gate_threshold=direction_gate_threshold,
            )

        self.regime_filter = RegimeFilter(
            calm_threshold=regime_calm_threshold,
            fear_threshold=regime_fear_threshold,
        )

        # Defensive Thresholds
        self.kill_switch_pct = 0.06
        self.is_killed = False
        self.drift_hurdle = 0.002 # Set to 0.2% as per your header note

        self.min_samples = min_samples
        self.weekend_crypto_adjustment = weekend_crypto_adjustment
        self.min_cash_buffer = min_cash_buffer
        self.lookback_days = 90
        self.market_proxy = "SPY"
        self.out_sample_start = pd.Timestamp('2025-07-01', tz='EST')

        if self.is_backtesting:
            self.tradeable_symbols = [yahoo_sym for _, yahoo_sym in ASSETS]
            self.crypto_symbols = ['BTC-USD']
        else:
            self.tradeable_symbols = [alpaca_sym for alpaca_sym, _ in ASSETS]
            self.crypto_symbols = ['BTC/USD']

        self.stock_symbols = [s for s in self.tradeable_symbols if s not in self.crypto_symbols]

    def _is_stock_market_closed(self):
        return not self.broker.is_market_open()

    def _is_market_open(self, symbol):
        if symbol in self.crypto_symbols:
            return True
        return self.broker.is_market_open()

    def _is_out_of_sample(self):
        return self.get_datetime() >= self.out_sample_start

    def _get_regime(self) -> dict:
        bars = self.get_historical_prices(self.market_proxy, 60, "day")
        if bars is None or len(bars) < 22:
            return {"regime": "CALM", "realised_vol": 0.15, "allocation_scale": 1.0}
        spy_close = bars.df["close"]
        return self.regime_filter.compute(spy_close, window=21)

    def _build_features_from_pipeline(self):
        today = self.get_datetime()
        end_date = today - pd.Timedelta(days=1)
        start_date = end_date - pd.Timedelta(days=self.lookback_days)
        raw_data = fetch_asset_data(ASSETS, self.is_backtesting, start_date, end_date)
        if not raw_data: return None, None, None, None
        featured = {sym: calculate_basic_features(df) for sym, df in raw_data.items()}
        enhanced = create_cross_asset_features(featured)
        ml_df = prepare_ml_features(enhanced)
        if ml_df.empty: return None, None, None, None
        X = ml_df.drop(columns=["symbol", "target_return", "target_volatility"])
        y_ret = ml_df["target_return"]; y_vol = ml_df["target_volatility"]
        return X, y_ret, y_vol, ml_df["symbol"]

    def on_trading_iteration(self):
        # --- TEMPORARY: FORCE CRYPTO TEST ---
        # if not hasattr(self, "_test_order_sent"):
        #     print("🚀 FORCING CRYPTO TEST ORDER: Verifying Alpaca Connectivity...")
        #
        #     from lumibot.entities import Asset
        #     base = "BTC"
        #     a = Asset(symbol=base, asset_type="crypto")
        #     q = Asset(symbol="USD", asset_type="forex") # USD is forex, not crypto
        #
        #     price = self.get_last_price(a, quote=q)
        #
        #     if price and price > 0:
        #         test_qty = round(100 / price, 4)
        #         limit_price = round(price * 1.005, 2)
        #
        #         order = self.create_order(
        #             a, test_qty, "buy", quote=q,
        #             type="limit", limit_price=limit_price, time_in_force="ioc"
        #         )
        #         self.submit_order(order)
        #         self._test_order_sent = True
        #         print(f"   ✅ Test Order Sent: Buy {test_qty} BTC @ ${limit_price} (Limit)")
        #     else:
        #         print("   ❌ ERROR: Could not fetch price using Asset objects.")
        # # ------------------------------------

        current_time = self.get_datetime()
        is_market_closed = self._is_stock_market_closed()
        portfolio_value = self.portfolio_value

        perf_record = {"timestamp": current_time, "value": portfolio_value}
        if self._is_out_of_sample():
            MLPortfolioStrategy.oos_tracker.append(perf_record)
        else:
            MLPortfolioStrategy.is_tracker.append(perf_record)

        print("\n" + "=" * 60)
        print(f"🔄 {current_time} | {'MARKET CLOSED' if is_market_closed else 'MARKET OPEN'}")
        print("=" * 60)

        try:
            if not hasattr(self, 'initial_capital'): self.initial_capital = portfolio_value
            if self.is_killed: return
            if portfolio_value <= self.initial_capital * (1 - self.kill_switch_pct):
                print(f"   🚨 KILL SWITCH! -{self.kill_switch_pct:.1%} breach.")
                for sym in self.tradeable_symbols:
                    pos = self.get_position(sym)
                    if pos and pos.quantity > 0: self.submit_order(self.create_order(sym, pos.quantity, "sell"))
                self.is_killed = True; return

            if is_market_closed and not self.weekend_crypto_adjustment: return
            regime_info = self._get_regime(); scale = regime_info["allocation_scale"]
            if scale == 0.0: return
            data = self._build_features_from_pipeline()
            if data[0] is None: return

            X, y_ret, y_vol, symbols = data
            mask = np.isfinite(y_ret.values) & np.isfinite(y_vol.values)
            X_c, yr_c, yv_c, s_c = X.loc[mask], y_ret.loc[mask], y_vol.loc[mask], symbols.loc[mask]

            if not self.load_pretrained: self.optimizer.fit(X_c, yr_c, yv_c)
            preds, latest_symbols = self.optimizer.predict_latest(X_c, s_c)

            target_weights = self.optimizer.optimal_weights(preds, latest_symbols, method="vol_parity")
            weights = {sym: max(0, w) for sym, w in target_weights.items()}
            total_w = sum(np.clip(list(weights.values()), 0, MAX_POSITION_PCT))
            weights = {s: (np.clip(weights.get(s, 0), 0, MAX_POSITION_PCT) / max(0.001, total_w)) * MAX_GROSS_EXPOSURE * scale for s in weights}

            if is_market_closed:
                adj = {}
                for sym, w in weights.items():
                    if sym in self.stock_symbols:
                        pos = self.get_position(sym); p = self.get_last_price(sym)
                        adj[sym] = (pos.quantity * p) / portfolio_value if (pos and p) else 0
                    else: adj[sym] = w
                weights = adj

            # ---- TWO-PHASE EXECUTION (Phase 1: Analysis) ----
            cash = self.get_cash(); avail = cash - (portfolio_value * self.min_cash_buffer)
            trade_list = []
            for sym, w in weights.items():
                if not self._is_market_open(sym): continue

                # CRITICAL UPDATE: Fetching live prices cleanly for crypto
                if sym in self.crypto_symbols and not self.is_backtesting:
                    from lumibot.entities import Asset
                    crypto_symbol = sym.split('/')[0]
                    crypto_asset = Asset(symbol=crypto_symbol, asset_type="crypto")
                    usd_asset = Asset(symbol="USD", asset_type="forex")
                    p = self.get_last_price(crypto_asset, quote=usd_asset)
                else:
                    p = self.get_last_price(sym)

                if not p or p <= 0: continue

                MLPortfolioStrategy.total_evaluations += 1

                target_q = round((w * portfolio_value) / p, 8 if sym in self.crypto_symbols else 4)
                current_q = self.get_position(sym).quantity if self.get_position(sym) else 0
                delta_q = target_q - current_q

                if abs(w - ((current_q * p) / portfolio_value)) < self.drift_hurdle:
                    MLPortfolioStrategy.skipped_evaluations += 1; continue

                trade_list.append({'sym': sym, 'dq': delta_q, 'p': p, 'dd': delta_q * p})

            # ---- PHASE 2: EXECUTE SELLS FIRST ----
            sorted_trades = sorted(trade_list, key=lambda x: x['dd'])
            orders_placed = 0

            for trade in sorted_trades:
                sym = trade['sym']; p = trade['p']; dq = trade['dq']; dd = trade['dd']
                if dq > 0 and dd > avail:
                    print(f"   ⚠️  {sym}: Scaling buy to {avail:,.2f}")
                    dq = avail / p; dd = dq * p
                if abs(dd) < MIN_TRADE_DOLLARS: continue

                side = "buy" if dq > 0 else "sell"
                l_p = round(p * (1.005 if side == "buy" else 0.995), 2)

                if sym in self.crypto_symbols and not self.is_backtesting:
                    from lumibot.entities import Asset
                    base = sym.split('/')[0]; a = Asset(symbol=base, asset_type="crypto")
                    q = Asset(symbol="USD", asset_type="forex")
                    order = self.create_order(a, abs(dq), side, quote=q, type="limit", limit_price=l_p, time_in_force="ioc")
                else:
                    order = self.create_order(sym, abs(dq), side, type="limit", limit_price=l_p, time_in_force="ioc")

                self.submit_order(order)
                MLPortfolioStrategy.total_spread_tax += (abs(dq) * p) * (self.expected_spreads.get(sym, 0.001) / 2)
                avail -= dd; orders_placed += 1
                print(f"   📝 {side.upper()} {abs(dq):.4f} {sym} @ ${p:.2f}")

        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            import traceback
            traceback.print_exc()

    def on_backtest_end(self):
        print("\n" + "=" * 60 + "\nBACKTEST COMPLETE: STATISTICAL METRICS\n" + "=" * 60)
        gross = MLPortfolioStrategy.oos_tracker[-1]["value"] if MLPortfolioStrategy.oos_tracker else self.portfolio_value
        tax = MLPortfolioStrategy.total_spread_tax
        print(f"GROSS TERMINAL VALUE:  ${gross:,.2f}")
        print(f"TOTAL SPREAD FRICTION: -${tax:,.2f}")
        print(f"NET TERMINAL VALUE:    ${gross - tax:,.2f}")
        print("-" * 60)

        if MLPortfolioStrategy.oos_tracker:
            df = pd.DataFrame(MLPortfolioStrategy.oos_tracker); equity = df["value"]; window = 45
            rol_ret = (equity / equity.shift(window)) - 1
            def get_mdd(w):
                m = w.cummax(); return ((w - m) / m).min()
            rol_dd = equity.rolling(window=window).apply(get_mdd)
            print(f"\n--- 45-DAY SPRINT METRICS (OOS) ---")
            print(f"Rolling Win Rate:      { (rol_ret.dropna() > 0).mean():.1%}")
            print(f"Prob. Ruin (>6% DD):   { (rol_dd.dropna() <= -self.kill_switch_pct).mean():.1%}")
            print(f"95th Pct Drawdown:     { rol_dd.dropna().quantile(0.05):.1%}")

        print("\n--- EXECUTION EFFICIENCY ---")
        total = MLPortfolioStrategy.total_evaluations
        skipped = MLPortfolioStrategy.skipped_evaluations

        if total > 0:
            skip_rate = skipped / total
            print(f"Total Trade Evaluations: {total}")
            print(f"Trades Skipped (Hurdle): {skipped}")
            print(f"Drift Hurdle Skip Rate:  {skip_rate:.1%}")
        else:
            print("No valid trades evaluated.")

        print("=" * 60)

    def _log_indicators(self, weights, regime_info, preds, latest_symbols):
        self.add_line("regime_scale", regime_info["allocation_scale"])
        self.add_line("portfolio_deployed", sum(weights.values()))