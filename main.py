import numpy as np
from environment import CartPoleEnv
from agent.a2c_agent import A2CAgent
from agent.dqn_agent import DQNAgent
from agent.ppo_agent import PPOAgent
from agent.base_agent import NeuralNetwork
from interface import Interface
import torch
import os
import warnings
warnings.filterwarnings("ignore")


def main():
    # Entraînement
    if interface.didtrain:
        agent.train(env, interface.episodes)
        print(f"Entraînement terminé pour {interface.episodes} épisodes.")

        # Sauvegarde
        if mode == "dqn":
            if interface.ask_save_dqn():
                os.makedirs(os.path.dirname(interface.path), exist_ok=True)
                torch.save(agent.nn.state_dict(), interface.path)

        elif mode == "a2c":
            if interface.ask_save_a2c():
                os.makedirs(os.path.dirname(interface.path), exist_ok=True)
                torch.save(agent.nna.state_dict(), interface.path.replace(".pth", "_actor.pth"))
                torch.save(agent.nnc.state_dict(), interface.path.replace(".pth", "_critic.pth"))
        
        elif mode == "ppo":
            if interface.ask_save_ppo():
                os.makedirs(os.path.dirname(interface.path), exist_ok=True)
                torch.save(agent.nna.state_dict(), interface.path.replace(".pth", "_actor.pth"))
                torch.save(agent.nnc.state_dict(), interface.path.replace(".pth", "_critic.pth"))
        
    # Test
    if interface.didtestfct():
        agent.test_agent(env, testepisodes=100)

    env.close()


if __name__ == "__main__":
    interface = Interface()
    env = CartPoleEnv()
    mode = interface.ask_mode()  # "dqn" or "a2c" or "ppo"

    if mode == "dqn":
        agent = DQNAgent(NeuralNetwork(), buffer_size=10000, batch_size=64, epsilon=0.9)

        if interface.ask_load_dqn():
            try:
                agent.nn.load_state_dict(torch.load(interface.path))
                print("Modèle DQN chargé.")
                interface.didtrainfct()
            except FileNotFoundError:
                print("Modèle DQN non trouvé, démarrage du programme...")
                interface.didtrain = True
                interface.episodes = int(input("How many episodes would you like to train the model for? "))
        else:
            interface.didtrain = True
            interface.episodes = int(input("How many episodes would you like to train the model for? "))

    elif mode == "a2c":
        agent = A2CAgent()

        if interface.ask_load_a2c():
            try:
                agent.nna.load_state_dict(torch.load(interface.path.replace(".pth", "_actor.pth")))
                agent.nnc.load_state_dict(torch.load(interface.path.replace(".pth", "_critic.pth")))
                print("Modèle A2C chargé.")
                interface.didtrainfct()
            except FileNotFoundError:
                print("Modèle A2C non trouvé, démarrage du programme...")
                interface.didtrain = True
                interface.episodes = int(input("How many episodes would you like to train the model for? "))
        else:
            interface.didtrain = True
            interface.episodes = int(input("How many episodes would you like to train the model for? "))

    elif mode == "ppo":
        agent = PPOAgent(buffer_size=1024)
        
        if interface.ask_load_ppo():
            try:
                agent.nna.load_state_dict(torch.load(interface.path.replace(".pth", "_actor.pth")))
                agent.nnc.load_state_dict(torch.load(interface.path.replace(".pth", "_critic.pth")))
                print("Modèle PPO chargé.")
                interface.didtrainfct()
            except FileNotFoundError:
                print("Modèle PPO non trouvé, démarrage du programme...")
                interface.didtrain = True
                interface.episodes = int(input("How many episodes would you like to train the model for? "))
        else:
            interface.didtrain = True
            interface.episodes = int(input("How many episodes would you like to train the model for? "))
    
    main()
