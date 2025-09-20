import numpy as np
import torch
from .base_agent import NeuralNetwork
from .base_agent import Common_Methods

class A2CAgent(Common_Methods):
    def __init__(self, hidden_dim=128, actor_lr=1e-3, critic_lr=1e-3, gamma=0.99):
        super().__init__(algo="A2C")
        self.nna = NeuralNetwork(hidden_dim=hidden_dim, output_dim=2, mode="actor", lr=actor_lr)  # Actor outputs probabilities for each action
        self.nnc = NeuralNetwork(hidden_dim=hidden_dim, output_dim=1, mode="critic", lr=critic_lr)  # Critic outputs a single value
        self.loss_fct = torch.nn.MSELoss()
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.gamma = gamma
        
    def getaction_a2c(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)  # Ajouter une dimension batch
        probs = self.nna(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.nnc(state)
        return action.item(), log_prob, value
    
    def update_a2c(self, rewards, log_probs, values, bootstrap_value):
        """next_state = torch.FloatTensor(next_state).unsqueeze(0)
        with torch.no_grad():
            next_value = self.nnc(next_state)  # Valeur du prochain état sans gradient
    
        target = rewards + (1 - dones) * self.gamma * next_value  # target = r + γV(s')"""
        
        T = rewards.shape[0]
        returns = torch.zeros(T, dtype=torch.float32)
        running = bootstrap_value.detach()
        for t in reversed(range(T)):
            running = rewards[t] + self.gamma * running
            returns[t] = running
            
        advantage = returns - values
        
        # Update Critic
        critic_loss = self.loss_fct(values, returns)
        self.nnc.optimizer.zero_grad()
        critic_loss.backward()
        self.nnc.optimizer.step()
        
        # Update Actor
        actor_loss = -(log_probs * advantage.detach()).mean()  # On détache l'avantage pour ne pas propager le gradient à travers le critic
        self.nna.optimizer.zero_grad()
        actor_loss.backward()
        self.nna.optimizer.step()