"""
Simple DQN agent with continuous actions
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
    
class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        # TODO consider experimenting with more layers / different nodes / different activations
        self.net = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_size)  # Q-values
        )

    def forward(self, x):
        return self.net(x)



class SimpleAgent:
    """DQN agent with continuous actions"""
    
    def __init__(self, state_size=4, action_size=3):
        self.state_size = state_size
        self.action_size = action_size
        
        # Device
        self.device = "cpu" # "cuda" is only possible on NVIDIA GPUs
        print(f"Using device: {self.device}")
        
        # Network
        self.model = DQNNetwork(state_size, action_size).to(self.device)
        # Since Adam's optimizer computes learning rates per parameter, it is efficient, 
        # robust and very suitable for DQN
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Memory
        self.memory = deque(maxlen=2000)

        # TODO experiment with different batch sizes
        self.batch_size = 32
        
        # Exploration
        # TODO experiment with lower minimum epsilon
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
    
    def act(self, state, training=True):
        """Choose action"""
        
        # Exploration
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        # Exploitation
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.model(state_tensor)
            return torch.argmax(q_values).item()
        
        return action.cpu().numpy()[0]
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience"""
        self.memory.append((state, action, reward, next_state, done))
    
    def train(self):
        """Train on batch from memory"""
        
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        
        states = torch.FloatTensor([x[0] for x in batch]).to(self.device)
        actions = torch.FloatTensor([x[1] for x in batch]).to(self.device)
        rewards = torch.FloatTensor([x[2] for x in batch]).to(self.device)
        next_states = torch.FloatTensor([x[3] for x in batch]).to(self.device)
        dones = torch.FloatTensor([x[4] for x in batch]).to(self.device)

        q_values = self.model(states)
        q_selected = q_values.gather(1, actions.long().unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.model(next_states)
            max_next_q = next_q_values.max(1)[0]
            targets = rewards + 0.99 * max_next_q * (1 - dones)

        loss = nn.MSELoss()(q_selected, targets)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, filepath):
        """Save model"""
        torch.save(self.model.state_dict(), filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model"""
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        print(f"Model loaded from {filepath}")


if __name__ == "__main__":
    agent = SimpleAgent()
    
    # Test forward pass
    dummy_state = np.array([0.01, 0.05, 0.02, 0.5])
    action = agent.act(dummy_state)
    
    print(f"Test action: {action}")