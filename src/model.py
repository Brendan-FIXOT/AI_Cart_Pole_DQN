import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class NeuralNetwork(nn.Module) :
    def __init__(self) :
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(4, 10)  # Couche d'entrée (4 entrées -> 10 neurones)
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