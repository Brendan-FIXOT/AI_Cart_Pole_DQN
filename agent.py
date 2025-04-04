import numpy as np
import torch
import torch.nn
import torch.optim as optim
from model import NeuralNetwork
from collections import deque
import random
from tqdm import tqdm

class Agent :
    def __init__(self, nn, buffer_size, batch_size, epsilon, epsilon_min=0.01, epsilon_max = 0.9, gamma=0.99) :
        self.nn = NeuralNetwork()
        self.epsilon = epsilon  # Probabilité d'exploration initiale
        self.epsilon_min = epsilon_min  # Valeur minimale d'epsilon
        self.epsilon_max = epsilon_max  # Valeur maximale d'epsilon
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        
        self.optimizer = optim.Adam(self.nn.parameters(), lr=0.001)
        
        self.loss_fct = torch.nn.MSELoss()
        
    def getaction(self, state) :
        if np.random.rand() < self.epsilon :
            return np.random.choice([0,1]) # exploration
        else : 
            with torch.no_grad() : # torch.no_grad pour éviter de taper dans la mémoire inutilement (car pas de backward ici)
                Q_values = self.nn.forward(state) # state déjà passer en tensor
            action = int(np.argmax(Q_values)) # Pas np.max, car ici on veut l'index (0 ou 1). Besoin du int, sinon action serait un tensor
            return action # Pas de rétropropagation tout de suite, car on veut la récompense associé à l'action
        
    def store_transition(self, state, action, reward, next_state, done) :
        self.memory.append((state, action, reward, next_state, done))
        
    def learn(self) :
        batch = random.sample(self.memory, self.batch_size)
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
        
        Q_values = self.nn(states).gather(1, actions) # self.nn(states) récupère les Qvalues associés à chaque choix (0 ou 1), ensuite le gather(1, actions) choisi la Qvalues par rapport à l'action choisie
        
        with torch.no_grad():
            max_next_Q = self.nn(next_states).max(1, keepdim=True)[0]  # Meilleure action future (keepdim = True permet de garder la forme (batch_size, 1))
            Q_targets = rewards + (1 - dones) * self.gamma * max_next_Q # Equation de Bellman calculant la Qvalues cible
            
        loss = self.loss_fct(Q_targets, Q_values) # Calcul de la perte en fonction des valeurs réelles (Q_values : valeurs prédite par le réseau de neuronne) et des meilleurs valeurs (Q_targets : valeurs maximal calculer avec l'eq de Bellman)
        
        self.optimizer.zero_grad() # Réinitialise les gradients
        loss.backward() # Rétropropagation
        self.optimizer.step() # Mise à jour des poids
        
    def train(self, env, episodes) :
        for episode in tqdm(range(episodes), desc="Entraînement", ncols=100, ascii=True):
        
            state = torch.tensor(env.reset(), dtype=torch.float32)
            done = False
            
            while not done :
                action = self.getaction(state)
                next_state, reward, done = env.step(action)
                
                next_state = torch.tensor(next_state, dtype=torch.float32)
                
                self.store_transition(state, action, reward, next_state, done) # Storage de la transition
                    
                if len(self.memory) > 1000 :
                    self.learn()
                
                state = next_state
                
            # Décroître epsilon à chaque épisode
            if self.epsilon > self.epsilon_min:
                self.epsilon = self.epsilon_max - (episode/episodes)  # Réduire epsilon progressivement
            #print(f"Épisode {episode + 1}/{episodes}, Epsilon: {self.epsilon:.4f}")
                
    
    def test_agent(self, env, testepisodes):
        total_rewards = []  # Liste pour enregistrer les récompenses totales obtenues par l'agent
        self.epsilon = 0

        for episode in range(testepisodes):
            state = torch.tensor(env.reset(), dtype=torch.float32)  # Convertir en tensor
            done = False
            total_reward = 0

            while not done :
                action = self.getaction(state)  # L'agent choisit une action avec la politique apprise
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