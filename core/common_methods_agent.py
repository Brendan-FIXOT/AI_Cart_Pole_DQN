import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from environment.cartpole_environment import CartPoleEnv
import imageio

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=128, output_dim=2, mode="dqn", optimizer=optim.Adam, lr=1e-3):
        super(NeuralNetwork, self).__init__()
        self.mode = mode
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.optimizer = optimizer(self.parameters(), lr=lr)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        if self.mode == "dqn" or self.mode == "critic" :
            return x  # Return raw Q-values // or state values
        elif self.mode == "actor":
            return F.softmax(x, dim=-1) # Apply softmax to get action probabilities
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

class Common_Methods :
    def __init__(self, algo):
        self.algo = algo  # on choisit "dqn" ou "A2C"
    
    # ===============================
    # Common methods
    # ===============================
    def train(self, env, episodes):
        for episode in tqdm(range(episodes), desc="Entraînement", ncols=100, ascii=True):
            state = env.reset()
            done = False

            if self.algo == "A2C":
                # reset buffers
                self.rewards, self.log_probs, self.values = [], [], []

            while not done:
                if self.algo == "dqn":
                    s = torch.tensor(state, dtype=torch.float32)
                    action = self.getaction_dqn(s)
                    next_state, reward, done = env.step(action)
                    next_state_t = torch.tensor(next_state, dtype=torch.float32)
                    self.store_transition_dqn(s, action, reward, next_state_t, done)
                    if len(self.memory) > 1000:
                        self.learn_dqn()
                    state = next_state

                elif self.algo == "A2C":
                    action, log_prob, value = self.getaction_a2c(state)
                    next_state, reward, done = env.step(action)

                    # stocker les infos de ce pas
                    self.rewards.append(float(reward))
                    self.log_probs.append(log_prob)
                    self.values.append(value)

                    state = next_state
                    
                elif self.algo == "ppo":
                    action, log_prob, value = self.getaction_ppo(state)
                    next_state, reward, done = env.step(action)

                    self.store_transition_ppo(
                        state,
                        action,
                        float(reward),
                        float(done),
                        log_prob.detach().squeeze(),
                        value.detach().squeeze())

                    self.state = next_state
                    
                    state = next_state
                    
                    # Update if buffer is full
                    if len(self.memory) >= self.buffer_size:
                        self.learn_ppo(state) # We pass the last state for bootstrap
                        
                    # No need to reset env here, just need coherence in state transition

            # fin d’épisode
            if self.algo == "dqn":
                if self.epsilon > self.epsilon_min:
                    self.epsilon = self.epsilon_max - (episode / episodes)

            elif self.algo == "A2C":
                if len(self.log_probs) > 0:  # sécurité
                    rewards_t   = torch.tensor(self.rewards, dtype=torch.float32)
                    log_probs_t = torch.stack(self.log_probs)
                    values_t    = torch.stack(self.values)
                    bootstrap_value = torch.tensor(0.0, dtype=torch.float32)  # CartPole -> fin terminale

                    self.update_a2c(rewards_t, log_probs_t, values_t, bootstrap_value)

            elif self.algo == "ppo":
                pass # Nothing to do here
            
                # reset buffers
                self.rewards, self.log_probs, self.values = [], [], []
                
    # ===============================
    # Test methods
    # ===============================
    def test_agent(self, env, testepisodes):
        total_rewards = []  # Liste pour enregistrer les récompenses totales obtenues par l'agent
        if self.algo == "dqn" :
            self.epsilon = 0 # exploitation seulement pendant les tests

        for episode in range(testepisodes):
            state = env.reset()
            done = False
            total_reward = 0

            while not done :
                if self.algo == "dqn" :
                    s = torch.tensor(state, dtype=torch.float32)
                    action = self.getaction_dqn(s)
                elif self.algo == "A2C" or self.algo == "ppo":
                    s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                    probs = self.nna(s).squeeze(0).detach().numpy()
                    action = int(np.argmax(probs))
                
                state, reward, done = env.step(action)
                total_reward += reward

            # Enregistrer la récompense totale pour cet épisode
            total_rewards.append(total_reward)
            print(f"Épisode {episode+1}/{testepisodes} - Récompense totale : {total_reward}")

        # Moyenne des récompenses sur tous les épisodes de test
        avg_reward = sum(total_rewards) / testepisodes
        print(f"\nRécompense moyenne sur {testepisodes} épisodes de test : {avg_reward}")
        
    def graphic_agent(self, filename):
        render_env = CartPoleEnv(render_mode="rgb_array")
        
        if self.algo == "dqn" :
            self.epsilon = 0 # exploitation seulement pendant les tests

        state = render_env.reset()
        done = False
        total_reward = 0
        frames = []

        # On estime un nombre max de frames pour la barre de progression (ex: 500)
        max_steps = 200
        with tqdm(total=max_steps, desc="Création GIF", ncols=100, ascii=True) as pbar:
            step = 0
            while not done and step < max_steps:
                render_env.render()
                if self.algo == "dqn" :
                    s = torch.tensor(state, dtype=torch.float32)
                    action = self.getaction_dqn(s)
                elif self.algo == "A2C" or self.algo == "ppo":
                    s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                    probs = self.nna(s).squeeze(0).detach().numpy()
                    action = int(np.argmax(probs))

                frame = render_env.render()
                frames.append(frame)
                state, reward, done = render_env.step(action)
                total_reward += reward
                step += 1
                pbar.update(1)
            
        render_env.close()
        imageio.mimsave(filename, frames, fps=30, loop=0)