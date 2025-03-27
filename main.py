import numpy as np
from environment import CartPoleEnv
from agent import Agent
from model import NeuralNetwork
import torch

def main() :
    env = CartPoleEnv()  # Initialiser l'environnement
    
    agent.train(env, episodes = 1000)
    
    torch.save(agent.nn.state_dict(), "model_checkpoint2.pth")
    print("Entraînement terminé et modèle sauvegardé.")
    
    if (didtest) :
        agent.test_agent(env, testepisodes = 10)

    env.close()  
    
                                                                                                                                                                                                                                                                                                                                                                                                                   
if __name__ == "__main__":
    nn = NeuralNetwork()
    agent = Agent(nn, buffer_size=10000, batch_size=64, epsilon=0.05)  # L'agent
    
    try:
        agent.nn.load_state_dict(torch.load("model_checkpoint2.pth"))
        print("Modèle chargé.")
    except FileNotFoundError:
        print("Modèle non trouvé, démarrage du programme...")
    
    didtest = True
        
    main()