# CartPole DQN (Deep Q-Network) Implementation

This repository contains an implementation of the **Deep Q-Network (DQN)** algorithm applied to the **CartPole** problem from OpenAI Gym. The project demonstrates the application of reinforcement learning using neural networks to solve the CartPole environment.

## Table of Contents
- [Project Overview](#project-overview)
- [Getting Started](#getting-started)
- [Dependencies](#dependencies)
- [Training the Agent](#training-the-agent)
- [Testing the Agent](#testing-the-agent)
- [Project Structure](#project-structure)
- [Results](#results)
- [License](#license)

## Project Overview

### Methodology

The **CartPole** problem involves balancing a pole on top of a moving cart. The goal is to apply forces to the cart in order to keep the pole upright for as long as possible. The agent learns to perform this task through reinforcement learning using the DQN algorithm.

The **Deep Q-Network (DQN)** is an off-policy reinforcement learning algorithm that uses a neural network to approximate the Q-value function. This implementation includes the following features:
- **Epsilon-greedy exploration** with a decaying epsilon value for more efficient exploration and exploitation.
- **Experience replay buffer** to store agent experiences and sample random batches for training, which improves learning efficiency.
- **Target network** to stabilize the learning process by using a target network to compute Q-values for the Bellman equation.

### Hyperparameters
We used the following hyperparameters for training the model:

| Hyperparameter       | Value            | Description |
|----------------------|------------------|-------------|
| **Learning Rate**     | 0.001            | The learning rate used for the Adam optimizer. |
| **Batch Size**        | 64               | Number of samples used per training batch. |
| **Epsilon**           | 0.05 (decaying)  | The epsilon parameter controls the exploration-exploitation tradeoff. It starts at 0.05 and decays over time as the agent becomes more confident. |
| **Gamma (Discount Factor)** | 0.99      | The discount factor used in the Q-learning algorithm. It represents the importance of future rewards. |
| **Buffer Size**       | 10,000           | The size of the replay buffer storing agent's experiences for training. |
| **Epochs**            | 1000             | The number of training episodes. |

### Performance Metrics
- **Total Reward**: Tracked during training to measure progress.

### Best model
- **model1000v4**: The average total reward for 100 tests was 517.6. Training consisted of 1000 episodes. There are other models where the training was longer, notably 3000 episodes, but the models were always worse than those of 1000 episodes. This is because the DQN without optimisation is quite unstable.

## Getting Started

### Prerequisites

Before running the code, ensure that you have Python installed on your system. You also need to have the following dependencies installed:

- Python 3.x
- PyTorch (for neural network training)
- OpenAI Gym (for the CartPole environment)
- NumPy (for numerical calculations)

### Installing Dependencies

To install the required dependencies, create a virtual environment and use the following:

```bash
pip install -r requirements.txt