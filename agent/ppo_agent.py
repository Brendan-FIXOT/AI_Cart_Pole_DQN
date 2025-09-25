import numpy as np
import torch
from collections import deque
from .base_agent import NeuralNetwork
from .base_agent import Common_Methods
import random

class PPOAgent(Common_Methods):
    def __init__(self, buffer_size=1024, hidden_dim=128, actor_lr=1e-3, critic_lr=1e-3, gamma=0.99, clip_value=0.2):
        super().__init__(algo="ppo")
        self.nna = NeuralNetwork(hidden_dim=hidden_dim, output_dim=2, mode="actor", lr=actor_lr)
        self.nnc = NeuralNetwork(hidden_dim=hidden_dim, output_dim=1, mode="critic", lr=critic_lr)
        self.loss_fct = torch.nn.MSELoss()
        self.buffer_size = buffer_size
        self.memory = deque(maxlen=self.buffer_size)
        self.clip_value = clip_value
        self.gamma = gamma
        
    def getaction_ppo(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)  # Ajouter une dimension batch
        probs = self.nna(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.nnc(state)
        return action.item(), log_prob, value

    def store_transition_ppo(self, state, action, reward, done, log_prob_old, value_old):
        self.memory.append((state, action, reward, done, log_prob_old, value_old))

    def learn_ppo(self):        
        states, actions, rewards, dones, old_log_probs, values = zip(*self.memory) # Learning on a complete rollout

        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
        old_log_probs = torch.stack(old_log_probs)
        values = torch.stack(values).squeeze()
        
        # Calculate returns
        returns = []
        R = 0
        for reward, done in zip(reversed(rewards), reversed(dones)): # We start from the end of the episode
            R = reward + self.gamma * R * (1 - done)
            returns.insert(0, R)   # we reconstruct from future to past
        returns = torch.tensor(returns, dtype=torch.float32)

        # Calculate advantages
        advantages = returns - values.detach()
        
        # New log probs and values
        probs = self.nna(states)
        dist = torch.distributions.Categorical(probs)
        new_log_probs = dist.log_prob(actions)
        new_values = self.nnc(states).squeeze()
        
        # PPO loss
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_value, 1 + self.clip_value) * advantages
        
        actor_loss = -torch.min(surr1, surr2).mean() # PPO objective (to maximize)
        critic_loss = self.loss_fct(new_values, returns)
        
        # Update networks with backpropagation
        self.nna.optimizer.zero_grad()
        actor_loss.backward()
        self.nna.optimizer.step()
        
        self.nnc.optimizer.zero_grad()
        critic_loss.backward()
        self.nnc.optimizer.step()
        
        # Do not forget to clear memory !!!
        self.memory.clear()

    def collector_trajectory(self, env, buffer_size):
        
        env.reset()
        
        for _ in range(buffer_size):
            action, log_prob, value = self.getaction_ppo(self.state)
            next_state, reward, done = env.step(action)

            self.store_transition_ppo(self.state, action, reward, done, log_prob, value)

            self.state = next_state
            
            if done:
                self.state = env.reset()