import numpy as np
from environment import CartPoleEnv
from agent import Agent
from model import NeuralNetwork
from interface import Interface
import torch

def main() :
    
    if (interface.didtrain) :
        agent.train(env, interface.episodes)
        torch.save(agent.nn.state_dict(), "model_saved/model_checkpoint2.pth")
        print(f"Entraînement terminé et modèle sauvegardé pour {interface.episodes} episodes.")
    
    if (interface.didtestfct()) :
        agent.test_agent(env, testepisodes = 10)

    env.close()  
    
                                                                                                                                                                                                                                                                                                                                                                                                                   
if __name__ == "__main__":
    nn = NeuralNetwork()
    agent = Agent(nn, buffer_size=10000, batch_size=64, epsilon=0.05)  # L'agent
    interface = Interface()
    env = CartPoleEnv()  # Initialiser l'environnement
    
    try:
        agent.nn.load_state_dict(torch.load("model_saved/model_checkpoint2.pth"))
        print("Modèle chargé.")
    except FileNotFoundError:
        print("Modèle non trouvé, démarrage du programme...")
    
    interface.didtrainfct()
    
    main()