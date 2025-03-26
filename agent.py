import numpy as np
import torch.nn
from model import NeuralNet
from collections import deque

class Agent :
    def __init__(self, nn, epsilon, buffer_size) :
        self.nn = NeuralNet()
        self.epsilon = epsilon  # Probabilité d'exploration
        self.memory = deque(maxlen=buffer_size)
        
    def getaction(self, state) :
        if np.random.rand() < self.epsilon :
            return np.random.choice([0,1]) # exploration
        else : 
            with torch.no_grad() : # torch.no_grad pour éviter de taper dans la mémoire inutilement (car pas de backward ici)
                Q_values = self.nn.forward(state)
            action = np.argmax(Q_values) # Pas np.max, car ici on veut l'index (0 ou 1)
            return action # Pas de rétropropagation tout de suite, car on veut la récompense associé à l'action
        
    def store_transition(self, state, action, reward, next_state, done) :
        self.memory.append((state, action, reward, next_state, done))
        
    def learn(self) :
        return None
        