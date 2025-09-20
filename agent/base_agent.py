import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

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

        if self.mode == "dqn":
            return x  # Return raw Q-values
        elif self.mode == "actor":
            return F.softmax(x, dim=-1) # Apply softmax to get action probabilities
        elif self.mode == "critic":
            return x # Return a scalar value for the state
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

class Common_Methods :
    def __init__(self, algo):
        self.algo = algo  # on choisit "dqn" ou "A2C"
    
    # ===============================
    # Common methods
    # ===============================
    def train(self, env, episodes) :
        for episode in tqdm(range(episodes), desc="Entraînement", ncols=100, ascii=True):
        
            state = torch.tensor(env.reset(), dtype=torch.float32)
            done = False
            
            while not done :
                if self.algo == "dqn" :
                    action = self.getaction_dqn(state)
                    next_state, reward, done = env.step(action)
                    next_state = torch.tensor(next_state, dtype=torch.float32)
                    self.store_transition_dqn(state, action, reward, next_state, done) # Storage de la transition
                    if len(self.memory) > 1000 :
                        self.learn_dqn()
                    state = next_state
                elif self.algo == "A2C" :
                    action, log_prob, value = self.getaction_a2c(state)
                    next_state, reward, done = env.step(action)
                    next_state = torch.tensor(next_state, dtype=torch.float32)
                    self.rewards.append(reward)
                    self.log_probs.append(log_prob)
                    self.values.append(value)
                    state = next_state
                    if done :
                        _, _, next_value = self.getaction_a2c(next_state) # Valeur du prochain état
                        self.update_a2c(torch.tensor(self.rewards, dtype=torch.float32), torch.stack(self.log_probs), torch.stack(self.values), next_value, torch.tensor(done, dtype=torch.float32), next_state)
                        self.rewards, self.log_probs, self.values = [], [], [] # Réinitialisation des listes pour le prochain épisode
                
            # Décroître epsilon à chaque épisode
            if self.algo == "dqn" and self.epsilon > self.epsilon_min:
                self.epsilon = self.epsilon_max - (episode / episodes)  # Réduire epsilon progressivement
            #print(f"Épisode {episode + 1}/{episodes}, Epsilon: {self.epsilon:.4f}")
                
    # ===============================
    # Test methods
    # ===============================
    def test_agent(self, env, testepisodes):
        total_rewards = []  # Liste pour enregistrer les récompenses totales obtenues par l'agent
        if self.algo == "dqn" :
            self.epsilon = 0 # exploitation seulement pendant les tests
        elif self.algo == "A2C" :
            pass # À implémenter plus tard

        for episode in range(testepisodes):
            state = torch.tensor(env.reset(), dtype=torch.float32)  # Convertir en tensor
            done = False
            total_reward = 0

            while not done :
                if self.algo == "dqn" :
                    action = self.getaction_dqn(state)  # L'agent choisit une action avec la politique apprise
                elif self.algo == "A2C" :
                    action, _, _ = self.getaction_a2c(state)  # L'agent choisit une action avec la politique apprise
                
                next_state, reward, done = env.step(action)
                next_state = torch.tensor(next_state, dtype=torch.float32)
                total_reward += reward 
                state = next_state

            # Enregistrer la récompense totale pour cet épisode
            total_rewards.append(total_reward)
            print(f"Épisode {episode+1}/{testepisodes} - Récompense totale : {total_reward}")

        # Moyenne des récompenses sur tous les épisodes de test
        avg_reward = sum(total_rewards) / testepisodes
        print(f"\nRécompense moyenne sur {testepisodes} épisodes de test : {avg_reward}")