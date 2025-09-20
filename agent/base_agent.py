import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

class NeuralNetwork(nn.Module) :
    def __init__(self) :
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(4, 10)  # Input layer (4 inputs -> 10 neurons)
        self.fc2 = nn.Linear(10, 10)  # Couche caché (10 -> 10 neuronnes)
        self.fc3 = nn.Linear(10,2)  # Couche de sortie (10 -> 2 choix)
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)

    def relu(self, x) :
        return F.relu(x)
    
    def forward(self, x) :
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Common_Methods :
    def __init__(self, algo="dqn"):
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
                elif self.algo == "A2C" :
                    pass # À implémenter plus tard
                
                state = next_state
                
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
                    pass # À implémenter plus tard
                
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