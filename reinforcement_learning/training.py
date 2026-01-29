"""
Simple training script for DQN
"""
import numpy as np
from data import get_btc_data
from trading_environment import SimpleTradingEnv
from agent import SimpleAgent


def train(episodes=100):
    """Train DQN agent"""
    
    print("Loading data...")
    data = get_btc_data("2020-01-01")
    
    # Split train/test
    split = int(len(data) * 0.8)
    train_data = data.iloc[:split]
    
    print(f"Training on {len(train_data)} days")
    
    # Create environment and agent
    env = SimpleTradingEnv(train_data)
    agent = SimpleAgent()
    
    # Track progress
    episode_rewards = []
    episode_values = []
    
    print(f"\nTraining for {episodes} episodes...")
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        steps = 0
        
        while True:
            # Agent chooses action
            action = agent.act(state, training=True)
            
            # Environment responds
            next_state, reward, done, info = env.step(action)
            
            # Agent remembers
            agent.remember(state, action, reward, next_state, done)
            
            # Agent learns
            agent.train()
            
            # Update
            state = next_state
            episode_reward += reward
            steps += 1
            
            if done:
                break
        
        # Track metrics
        episode_rewards.append(episode_reward)
        episode_values.append(info['portfolio_value'])
        
        # Log every 10 episodes
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_value = np.mean(episode_values[-10:])
            
            print(
                f"Episode {episode + 1}/{episodes} | "
                f"Avg Reward: {avg_reward:.4f} | "
                f"Avg Portfolio: ${avg_value:,.0f} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )
    
    # Save trained agent
    agent.save("models/simple_dqn.pth")
    
    print("\nTraining complete!")
    print(f"Final portfolio value: ${episode_values[-1]:,.2f}")
    
    return agent


if __name__ == "__main__":
    import os
    os.makedirs("models", exist_ok=True)
    
    train(episodes=100)