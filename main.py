import numpy as np
from environment import CartPoleEnv
from agent import Agent
from model import NeuralNetwork
import torch

def main():
    for episode in range(1000):
        
        state = torch.tensor(env.reset(), dtype=torch.float32)
        done = False
        
        while not done :
            action = agent.getaction(state)  # Action aléatoire
            next_state, reward, done = env.step(action)
            
            next_state = torch.tensor(next_state, dtype=torch.float32)
            
            agent.store_transition(state, action, reward, next_state, done) # Storage de la transition
                
            if len(agent.memory) > 1000 :
                agent.learn()
            
            state = next_state
            
    torch.save(agent.nn.state_dict(), "model_checkpoint.pth")
    print("Entraînement terminé et modèle sauvegardé.")

    env.close()  
    
                                                                                                                                                                                                                                                                                                                                                                                                                   
if __name__ == "__main__":
    nn = NeuralNetwork()
    env = CartPoleEnv()  # Initialiser l'environnement
    agent = Agent(nn, epsilon=0.05, buffer_size=10000, batch_size=64)  # L'agent
    torch.load(agent.nn.state_dict(), "model_checkpoint.pth")
    main()