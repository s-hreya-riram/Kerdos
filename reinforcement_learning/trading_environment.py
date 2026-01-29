"""
Simple crypto trading environment
"""
import gym
from gym import spaces
import numpy as np


class SimpleTradingEnv(gym.Env):
    """
    Simple trading environment with discrete actions
    
    State: [returns, price_to_ma20, volatility, current_position]
    Action: Target position from 0 (no crypto) to 1 (all-in crypto)
    Reward: Portfolio return

    TODO rename references from "crypto" to "asset" for generalization
    """
    
    def __init__(self, data, initial_balance=100000):
        super().__init__()
        
        self.data = data.reset_index(drop=True)
        self.initial_balance = initial_balance
        
        # Start after warm-up period
        self.start_step = 50
        self.current_step = self.start_step
        
        # Portfolio state
        self.balance = initial_balance
        self.crypto_held = 0.0
        self.portfolio_value = initial_balance
        
        # Actions: 0 = 0%, 1 = 50%, 2 = 70%
        # TODO consider continuous allocation instead of discrete
        self.action_space = spaces.Discrete(3)
        
        # Observation: [returns_5d, price_to_ma20, volatility, position]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )
    
    def reset(self):
        """Start new episode"""
        self.current_step = self.start_step
        self.balance = self.initial_balance
        self.crypto_held = 0.0
        self.portfolio_value = self.initial_balance
        
        return self._get_state()
    
    def _get_state(self):
        """Build state vector"""
        row = self.data.iloc[self.current_step]
        
        # Features
        returns_5d = row['returns'] * 5  # Approximate 5-day
        price_to_ma20 = row['price_to_ma20']
        volatility = row['volatility']
        
        # Current position
        current_price = row['Close']
        crypto_value = self.crypto_held * current_price
        self.portfolio_value = self.balance + crypto_value
        
        if self.portfolio_value > 0:
            position = crypto_value / self.portfolio_value
        else:
            position = 0
        
        state = np.array([
            returns_5d,
            price_to_ma20,
            volatility,
            position
        ], dtype=np.float32)
        
        return state
    
    def step(self, action):
        """Execute action and return (state, reward, done, info)"""
        
        # Get current price
        current_price = self.data.iloc[self.current_step]['Close']
        
        # Target allocation based on action
        allocation_map = [0.0, 0.5, 0.7]
        target_allocation = allocation_map[int(action)]

        
        # Calculate target value
        crypto_value = self.crypto_held * current_price
        self.portfolio_value = self.balance + crypto_value
        
        target_crypto_value = self.portfolio_value * target_allocation
        
        # Trade (simplified, no transaction costs for now)
        trade_value = target_crypto_value - crypto_value
        
        if trade_value > 0:  # Buying
            if trade_value <= self.balance:
                self.crypto_held += trade_value / current_price
                self.balance -= trade_value
        else:  # Selling
            sell_crypto = abs(trade_value) / current_price
            if sell_crypto <= self.crypto_held:
                self.crypto_held -= sell_crypto
                self.balance += abs(trade_value)
        
        # Move to next day
        self.current_step += 1
        
        # Calculate reward (daily return)
        next_price = self.data.iloc[self.current_step]['Close']
        new_crypto_value = self.crypto_held * next_price
        new_portfolio_value = self.balance + new_crypto_value
        
        reward = (new_portfolio_value - self.portfolio_value) / self.portfolio_value
        self.portfolio_value = new_portfolio_value
        
        # Check if done
        done = self.current_step >= len(self.data) - 1
        
        # Get next state
        next_state = self._get_state() if not done else np.zeros(4)
        
        info = {
            'portfolio_value': self.portfolio_value,
            'position': target_allocation
        }
        
        return next_state, reward, done, info


if __name__ == "__main__":
    from data import get_btc_data
    
    # Test environment
    data = get_btc_data("2023-01-01")
    env = SimpleTradingEnv(data)
    
    state = env.reset()
    print(f"Initial state: {state}")
    
    # Random action
    action = env.action_space.sample()
    next_state, reward, done, info = env.step(action)
    
    print(f"Action: {action}")
    print(f"Reward: {reward}")
    print(f"Portfolio: ${info['portfolio_value']:,.2f}")