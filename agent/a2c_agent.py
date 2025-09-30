import numpy as np
import torch
from core.common_methods_agent import NeuralNetwork
from core.common_methods_agent import Common_Methods

class A2CAgent(Common_Methods):
    def __init__(self, input_dim=4, hidden_dim=128, actor_lr=1e-3, critic_lr=1e-3, gamma=0.99):
        super().__init__(algo="a2c")
        if torch.cuda.is_available(): # CUDA NVIDIA
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():  # MAC M1/M2/M3
            self.device = torch.device("mps")
        #elif torch.version.hip is not None:     # AMD ROCm
            #self.device = torch.device("hip") # Uniquement sur Linux
        else:
            self.device = torch.device("cpu")
        self.nna = NeuralNetwork(hidden_dim=hidden_dim, input_dim=input_dim, output_dim=2, mode="actor", lr=actor_lr)  # Actor outputs probabilities for each action
        self.nnc = NeuralNetwork(hidden_dim=hidden_dim, input_dim=input_dim, output_dim=1, mode="critic", lr=critic_lr)  # Critic outputs a single value
        self.nna.to(self.device)
        self.nnc.to(self.device)
        self.loss_fct = torch.nn.MSELoss()
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.gamma = gamma
    
    def getaction_a2c(self, state):
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)  # Ajouter une dimension batch
        probs = self.nna(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.nnc(state)
        return action.item(), log_prob, value
    
    def update_a2c(self, rewards, log_probs, values, bootstrap_value):
        T = rewards.shape[0]
        returns = torch.zeros(T, dtype=torch.float32, device=self.device)
        running = bootstrap_value.detach()
        for t in reversed(range(T)):
            running = rewards[t] + self.gamma * running
            returns[t] = running
            
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Update Critic
        critic_loss = self.loss_fct(values, returns.detach())
        self.nnc.optimizer.zero_grad()
        critic_loss.backward()
        self.nnc.optimizer.step()
        
        # Update Actor
        actor_loss = -(log_probs * advantages.detach()).mean()  # On détache l'avantage pour ne pas propager le gradient à travers le critic
        self.nna.optimizer.zero_grad()
        actor_loss.backward()
        self.nna.optimizer.step()