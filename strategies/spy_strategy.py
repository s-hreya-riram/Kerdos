import pandas as pd
from lumibot.strategies.strategy import Strategy

from data.constants import MAX_GROSS_EXPOSURE

class SPYBenchmarkStrategy(Strategy):

    def __init__(self, broker, performance_callback=None, invest_ratio=MAX_GROSS_EXPOSURE, **kwargs):
        super().__init__(broker, **kwargs)

        print("🔧 INIT: Creating SPYBenchmarkStrategy")
        print(f"   performance_callback provided: {performance_callback is not None}")

        self.sleeptime = "1D"
        self.symbol = "SPY"
        self.invest_ratio = invest_ratio
        self.has_bought = False

        if performance_callback is None:
            performance_callback = self.parameters.get("performance_callback")

        self.performance_callback = performance_callback

        print(f"   self.performance_callback set: {self.performance_callback is not None}")

        self.market_proxy = "SPY"

        self.in_sample_end = pd.Timestamp('2025-06-30', tz='EST')
        self.out_sample_start = pd.Timestamp('2025-07-01', tz='EST')

        self.in_sample_performance = []
        self.out_sample_performance = []

        print("📋 Benchmark symbol:", self.symbol)

    # -------------------------------------------------

    def _is_out_of_sample(self):
        now = self.get_datetime()
        return now >= self.out_sample_start

    # -------------------------------------------------

    def on_trading_iteration(self):

        print("\n" + "="*60)
        print(f"🔄 SPY on_trading_iteration() CALLED at {self.get_datetime()}")
        print(f"   callback exists: {self.performance_callback is not None}")
        print("="*60)

        try:
            print("   📍 ENTERING try block...")

            # ===== Buy & Hold Logic =====
            if not self.has_bought:

                cash = self.get_cash()
                price = self.get_last_price(self.symbol)

                print(f"   💵 Cash available: {cash}")
                print(f"   📈 SPY price: {price}")

                if cash > 0 and price and price > 0:

                    quantity = (cash * self.invest_ratio) / price

                    if quantity > 0:
                        order = self.create_order(self.symbol, quantity, "buy")
                        self.submit_order(order)

                        self.has_bought = True
                        print(f"   ✅ Executed SPY buy: {quantity:.4f} shares")

            # ===== Track performance =====
            portfolio_value = self.get_portfolio_value()

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

    # -------------------------------------------------

    def _trigger_performance_callback(self):

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

            # ----- Collect positions -----
            positions_dict = {}
            positions = self.get_positions()

            if positions:
                for pos in positions:
                    symbol = pos.symbol
                    qty = pos.quantity

                    if abs(qty) > 0.001:
                        price = self.get_last_price(symbol)
                        if price:
                            positions_dict[symbol] = {
                                'quantity': float(qty),
                                'value': float(abs(qty * price)),
                                'avg_price': float(price)
                            }

            print(f"      📦 Positions: {len(positions_dict)}")

            # ----- Benchmark has no predictions -----
            performance_data = {
                'timestamp': current_datetime,
                'portfolio_value': float(portfolio_value),
                'cash': float(cash),
                'positions': positions_dict,
                'is_out_of_sample': self._is_out_of_sample(),
                'weights': {},
                'predictions': {
                    'returns': [],
                    'volatility': [],
                    'symbols': []
                }
            }

            print(f"      📞 CALLING callback function...")
            self.performance_callback(performance_data)
            print(f"      ✅ Callback returned successfully")

        except Exception as e:
            print(f"      ❌ Performance callback error: {e}")
            import traceback
            traceback.print_exc()

    # -------------------------------------------------

    def on_backtest_end(self):

        print("\n" + "="*60)
        print("SPY BENCHMARK BACKTEST COMPLETE")
        print("="*60)

        if self.in_sample_performance:
            df = pd.DataFrame(self.in_sample_performance)
            ret = (df['value'].iloc[-1] / df['value'].iloc[0] - 1) * 100
            print(f"IN-SAMPLE RETURN: {ret:.2f}%")

        if self.out_sample_performance:
            df = pd.DataFrame(self.out_sample_performance)
            ret = (df['value'].iloc[-1] / df['value'].iloc[0] - 1) * 100
            print(f"OUT-OF-SAMPLE RETURN: {ret:.2f}%")

        print("="*60 + "\n")
