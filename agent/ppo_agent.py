import numpy as np
import torch
from collections import deque
from core.common_methods_agent import NeuralNetwork
from core.common_methods_agent import Common_Methods
import random

class PPOAgent(Common_Methods):
    def __init__(self, buffer_size=512, batch_size=64, nb_epochs=4, input_dim=4, hidden_dim=128, actor_lr=1e-3, critic_lr=1e-3, gamma=0.99, clip_value=0.2, lambda_gae=0.95, entropy_bonus=False, shuffle=True):
        super().__init__(algo="ppo")
        self.nna = NeuralNetwork(hidden_dim=hidden_dim, input_dim=input_dim, output_dim=2, mode="actor", lr=actor_lr)
        self.nnc = NeuralNetwork(hidden_dim=hidden_dim, input_dim=input_dim, output_dim=1, mode="critic", lr=critic_lr)
        self.loss_fct = torch.nn.MSELoss()
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.nb_epochs = nb_epochs
        self.memory = deque(maxlen=self.buffer_size)
        self.gamma = gamma
        self.clip_value = clip_value
        self.lambda_gae = lambda_gae
        self.c1 = 0.5
        self.c2 = 0.01
        self.ent_bonus = entropy_bonus
        self.shuffle = shuffle
        
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
        
    def compute_gae(self, rewards, values, dones, next_value):
        T = len(rewards)
        advantages = torch.zeros(T, dtype=torch.float32)
        
        gae = 0.0
        values = torch.cat((values, torch.tensor([next_value], dtype=torch.float32)))
        
        for t in reversed(range(T)):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lambda_gae * gae * (1 - dones[t])
            advantages[t] = gae
            
        returns = advantages + values[:-1] # R_t = A_t + V(s_t)
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) # Normalization
        return advantages, returns
        
    def learn_ppo(self, last_state):        
        states, actions, rewards, dones, old_log_probs, values = zip(*self.memory) # Learning on a complete rollout

        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32) # unsqueeze not needed, already 1D for the compute_gae, dones and rewards are not used in the loss directly
        old_log_probs = torch.stack(old_log_probs)
        values = torch.stack(values).squeeze()
        
        last_state = torch.as_tensor(last_state, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            last_value = self.nnc(last_state).squeeze(-1)

        next_value = last_value * (1.0 - dones[-1]) # if last state is done, so last value is 0
        advantages, returns = self.compute_gae(rewards, values, dones, next_value) # Bootstrap value for the last state
        
        for epoch in range(self.nb_epochs):
            # indices shuffle or not
            if self.shuffle:
                idx = torch.randperm(self.buffer_size)
            else:
                idx = torch.arange(self.buffer_size)
                
            for start in range(0, self.buffer_size, self.batch_size):
                end = start + self.batch_size
                batch_idx = idx[start:end]

                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]

                # New log probs and values
                probs = self.nna(batch_states)
                dist = torch.distributions.Categorical(probs)
                new_log_probs = dist.log_prob(batch_actions)
                new_values = self.nnc(batch_states).squeeze()

                # PPO loss
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages.detach()
                surr2 = torch.clamp(ratio, 1 - self.clip_value, 1 + self.clip_value) * batch_advantages.detach()

                # Optional: entropy bonus for exploration
                if self.ent_bonus:
                    entropy = dist.entropy().mean()
                    actor_loss = -torch.min(surr1, surr2).mean() - self.c1 * entropy
                    critic_loss = self.c2 * self.loss_fct(new_values, batch_returns)
                else:
                    actor_loss = -torch.min(surr1, surr2).mean()
                    critic_loss = self.c1 * self.loss_fct(new_values, batch_returns)

                # Update networks with backpropagation
                self.nna.optimizer.zero_grad()
                actor_loss.backward()
                self.nna.optimizer.step()

                self.nnc.optimizer.zero_grad()
                critic_loss.backward()
                self.nnc.optimizer.step()
        
        # Do not forget to clear memory
        self.memory.clear()