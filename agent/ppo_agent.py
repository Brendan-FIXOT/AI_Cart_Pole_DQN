import numpy as np
import torch
from .base_agent import NeuralNetwork
from .base_agent import Common_Methods

class PPOAgent(Common_Methods):
    def __init__(self, buffer_size=1024, batch_size=64, hidden_dim=128, actor_lr=1e-3, critic_lr=1e-3, gamma=0.99, clip_value=0.2):
        super().__init__(algo="ppo")
        self.nna = NeuralNetwork(hidden_dim=hidden_dim, output_dim=2, mode="actor", lr=actor_lr)
        self.nnc = NeuralNetwork(hidden_dim=hidden_dim, output_dim=1, mode="critic", lr=critic_lr)
        self.loss_fct = torch.nn.MSELoss()
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.clip_value = clip_value
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.gamma = gamma
        
    def getaction_ppo(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)  # Ajouter une dimension batch
        probs = self.nna(state)
        dist = torch.distributions.Categorical(probs)
        log_prob = dist.log_prob(action)
        action = dist.sample()
        value = self.nnc(state)
        return action.item(), log_prob, value

    def store_transition_ppo(self, state, action, reward, done, log_prob_old, value_old):
        self.memory.append((state, action, reward, done, log_prob_old, value_old))

    def learn_ppo(self):
        batch = random.sample(self.memory, self.batch_size)
        
        states, actions, rewards, dones, log_probs, values = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
        log_probs = torch.stack(log_probs).unsqueeze(1)
        values = torch.stack(values).unsqueeze(1)
        
        # Calculate returns and advantages
        returns = rewards + self.gamma * next_states * (1 - dones) - values # Discounted returns sum
        advantages = returns - values
        
        for state in zip(states, actions, log_probs, advantages):
            _, new_log_prob, new_value = self.getaction_ppo(state)
        
        ratio = torch.exp(new_log_prob - log_prob)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_value, 1 + self.clip_value) * advantages
        actor_loss = -torch.min(surr1, surr2).mean() # PPO objective (to maximize)
        critic_loss = self.loss_fct(new_value, returns)
        
        self.nna.optimizer.zero_grad()
        actor_loss.backward()
        self.nna.optimizer.step()
        
        self.nnc.optimizer.zero_grad()
        critic_loss.backward()
        self.nnc.optimizez.step()

    def collector_trajectory(self, env, buffer_size):
        
        env.reset()
        
        for _ in range(buffer_size):
            action, log_prob, value = self.getaction_ppo(self.state)
            next_state, reward, done = env.step(action)

            self.store_transition_ppo(self.state, action, reward, done, log_prob, value)

            self.state = next_state
            
            if done:
                self.state = env.reset()