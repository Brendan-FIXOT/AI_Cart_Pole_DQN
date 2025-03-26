import numpy as np
from environment import CartPoleEnv
from agent import Agent
from model import NeuralNetwork

def main():
    env = CartPoleEnv(render_mode=None)
    
    nn = NeuralNetwork()
    
    agent = Agent(nn, 0.3, 10000)
    
    state, _ = env.reset()
    
    for _ in range(1000):
        action = agent.getaction(state)  # Action aléatoire
        next_state, reward, done = env.step(action)
        
        agent.store_transition(state, action, reward, next_state, done) # Storage de la transition
        
        if done:
            env.reset()
        else :
            state = next_state
            
        if len(agent.memory) > 1000 :
            agent.learn()

    env.close()
    
main()