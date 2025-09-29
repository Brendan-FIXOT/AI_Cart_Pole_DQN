import numpy as np
import torch
from collections import deque
from core.common_methods_agent import NeuralNetwork
from core.common_methods_agent import Common_Methods
import random

class PPOAgent(Common_Methods):
    def __init__(self, buffer_size=512, input_dim=4, hidden_dim=128, actor_lr=1e-3, critic_lr=1e-3, gamma=0.99, clip_value=0.2):
        super().__init__(algo="ppo")
        self.nna = NeuralNetwork(hidden_dim=hidden_dim, input_dim=input_dim, output_dim=2, mode="actor", lr=actor_lr)
        self.nnc = NeuralNetwork(hidden_dim=hidden_dim, input_dim=input_dim, output_dim=1, mode="critic", lr=critic_lr)
        self.loss_fct = torch.nn.MSELoss()
        self.buffer_size = buffer_size
        self.memory = deque(maxlen=self.buffer_size)
        self.clip_value = clip_value
        self.gamma = gamma
        
    @torch.no_grad() # We don't want to compute gradients when selecting actions, because we are not training
    def getaction_ppo(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.nna(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.nnc(state)
        return action.item(), log_prob, value

    def store_transition_ppo(self, state, action, reward, done, log_prob_old, value_old):
        self.memory.append((state, action, reward, done, log_prob_old, value_old))
        
    def compute_gae(self, rewards, values, dones, next_value, lam=0.95):
        T = len(rewards)
        advantages = torch.zeros(T, dtype=torch.float32)
        
        gae = 0.0
        values = torch.cat((values, torch.tensor([next_value], dtype=torch.float32)))
        
        for t in reversed(range(T)):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * lam * gae * (1 - dones[t])
            advantages[t] = gae
            
        returns = advantages + values[:-1] # R_t = A_t + V(s_t)
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) # Normalization
        return advantages, returns
        
    def learn_ppo(self):        
        states, actions, rewards, dones, old_log_probs, values = zip(*self.memory) # Learning on a complete rollout

        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
        old_log_probs = torch.stack(old_log_probs)
        values = torch.stack(values).squeeze()
        
        # Compute returns and advantages
        advantages, returns = self.compute_gae(rewards, values, dones, next_value=0)
        
        # New log probs and values
        probs = self.nna(states)
        dist = torch.distributions.Categorical(probs)
        new_log_probs = dist.log_prob(actions)
        new_values = self.nnc(states).squeeze()
        
        # PPO loss
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages.detach()
        surr2 = torch.clamp(ratio, 1 - self.clip_value, 1 + self.clip_value) * advantages.detach()
        
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