import numpy as np
import torch
import torch.nn
import torch.optim as optim
from model import NeuralNetwork
from collections import deque
import random

class Agent :
    def __init__(self, nn, epsilon, buffer_size, batch_size) :
        self.nn = NeuralNetwork()
        self.epsilon = epsilon  # Probabilité d'exploration
        self.memory = deque(maxlen=buffer_size)
        loss_fct = torch.nn.MSELoss()
        
    def getaction(self, state) :
        if np.random.rand() < self.epsilon :
            return np.random.choice([0,1]) # exploration
        else : 
            with torch.no_grad() : # torch.no_grad pour éviter de taper dans la mémoire inutilement (car pas de backward ici)
                state_tensor = torch.tensor(state, dtype=torch.float32)  # Conversion en Tensor
                Q_values = self.nn.forward(state_tensor)
            action = int(np.argmax(Q_values)) # Pas np.max, car ici on veut l'index (0 ou 1). Besoin du int, sinon action serait un tensor
            return action # Pas de rétropropagation tout de suite, car on veut la récompense associé à l'action
        
    def store_transition(self, state, action, reward, next_state, done) :
        self.memory.append((state, action, reward, next_state, done))
        
    def learn(self) :
        batch = random(self.memory, self.batch_size)
        
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