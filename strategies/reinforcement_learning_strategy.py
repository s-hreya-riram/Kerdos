"""
Simple DQN strategy for demo
"""
from lumibot.strategies import Strategy
import numpy as np
import torch
import sys
sys.path.append('.')  # Make sure we can import from rl/

from reinforcement_learning.agent import SimpleAgent


class Strategy(Strategy):
    """
    Simple DQN trading strategy
    """
    
    def initialize(self):
        self.sleeptime = "1D"
        
        # Load trained agent
        self.agent = SimpleAgent()
        try:
            self.agent.load("models/simple_dqn.pth")
            self.agent.epsilon = 0  # No exploration in live trading
            print("✅ DQN agent loaded")
        except:
            print("❌ Failed to load agent")
            self.agent = None
        
        # Risk limits
        self.max_position = 0.70
    
    # TODO expand to multiple assets, move asset list to config in utils to share with data.py
    def on_trading_iteration(self):
        """Main trading logic"""
        self.log_message("on_trading_iteration fired")
        
        if self.agent is None:
            return
        
        symbol = "BTC-USD"
        
        # Build state
        state = self._build_state(symbol)
        
        if state is None:
            return
        
        # Get RL action (target allocation)
        action = self.agent.act(state, training=False)
        allocation_map = [0.0, 0.5, 0.7]  # respect risk cap
        target_allocation = allocation_map[action]

        # Execute
        self._set_allocation(symbol, target_allocation)
        
        self.log_message(
        f"RL action={action}, "
        f"target_alloc={target_allocation:.2%}, "
        f"PV={self.get_portfolio_value():,.2f}"
)

    
    def _build_state(self, symbol):
        """Build state vector"""
        try:
            # Get 100 days of history
            bars = self.get_historical_prices(symbol, 100, "1D")
            
            if bars is None or len(bars.df) < 50:
                return None
            
            df = bars.df
            
            # Calculate features (match training)
            df['returns'] = df['close'].pct_change()
            df['ma_20'] = df['close'].rolling(20).mean()
            df['price_to_ma20'] = df['close'] / df['ma_20'] - 1
            df['volatility'] = df['returns'].rolling(20).std()
            
            # Get current values
            current = df.iloc[-1]
            
            returns_5d = current['returns'] * 5
            price_to_ma20 = current['price_to_ma20']
            volatility = current['volatility']
            
            # Current position
            position = self.get_position(symbol)
            portfolio_value = self.get_portfolio_value()
            
            if position and portfolio_value > 0:
                crypto_value = position.quantity * self.get_last_price(symbol)
                position_pct = crypto_value / portfolio_value
            else:
                position_pct = 0
            
            state = np.array([
                returns_5d,
                price_to_ma20,
                volatility,
                position_pct
            ], dtype=np.float32)
            
            return state
            
        except Exception as e:
            print(f"Error building state: {e}")
            return None
    
    def _set_allocation(self, symbol, target_allocation):
        """Set position to target allocation"""
        
        portfolio_value = self.get_portfolio_value()
        last_price = self.get_last_price(symbol)
        
        if last_price is None or last_price <= 0:
            return
        
        # Target quantity
        target_value = portfolio_value * target_allocation
        target_qty = target_value / last_price
        
        # Current quantity
        position = self.get_position(symbol)
        current_qty = position.quantity if position else 0
        
        # Calculate trade
        delta = target_qty - current_qty
        
        if abs(delta) < 0.001:  # Too small, skip
            return


        # Execute
        side = "buy" if delta > 0 else "sell"

        self.log_message(
            f"Submitting order: {side}, qty={abs(delta):.6f}"
        )
        order = self.create_order(symbol, abs(delta), side)
        self.submit_order(order)