import torch
import torch.nn as nn
import torch.optim as optim

class NeuralNetwork(nn.Module) :
    def __init__(self) :
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(4, 10)  # Couche d'entrée (4 entrées -> 10 neurones)
        self.fc2 = nn.Linear(10, 10)  # Couche caché (10 -> 10 neuronnes)
        self.fc3 = nn.Linear(10,2)  # Couche de sortie (10 -> 2 choix)
        self.optimizer = optim.Adam(nn.parameters(), lr=0.01)

    def forward(self, x) :
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
 
    def relu(self, x) :
        return nn.ReLU(x)
    
    def backward(self, Y_true, Y_pred) :
        loss = nn.MSELoss(Y_true, Y_pred)
        loss.backward()
        self.optimizer.step()