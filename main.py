import numpy as np
from environment import CartPoleEnv
from agent import Agent
from model import NeuralNetwork
from interface import Interface
import torch
import warnings
warnings.filterwarnings("ignore")


def main() :
    
    if (interface.didtrain) :
        agent.train(env, interface.episodes)
        print(f"Entraînement terminé pour {interface.episodes} episodes.")
        if (interface.ask_save()) :
            torch.save(agent.nn.state_dict(), f"{interface.path}")
    
    if (interface.didtestfct()) :
        agent.test_agent(env, testepisodes = 100)

    env.close()  
    
                                                                                                                                                                                                                                                                                                                                                                                                                   
if __name__ == "__main__":
    nn = NeuralNetwork()
    agent = Agent(nn, buffer_size=10000, batch_size=64, epsilon=0.9)  # L'agent
    interface = Interface()
    env = CartPoleEnv()  # Initialiser l'environnement
    
    if (interface.ask_load()) :
        try:
            agent.nn.load_state_dict(torch.load(f"{interface.path}"))
            print("Modèle chargé.")
            interface.didtrainfct()
        except FileNotFoundError:
            print("Modèle non trouvé, démarrage du programme...")
            interface.didtrain = True # On lance l'entraînement d'un nouveau modèle
            interface.episodes = int(input("How many episodes would you like to train the model for? "))
    else :
        interface.didtrain = True # On lance l'entraînement d'un nouveau modèle
        interface.episodes = int(input("How many episodes would you like to train the model for? "))
    
    main()