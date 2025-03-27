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

The **CartPole** problem involves balancing a pole on top of a moving cart. The goal is to apply forces to the cart in order to keep the pole upright for as long as possible. The agent learns to perform this task through reinforcement learning using the DQN algorithm.

The **Deep Q-Network (DQN)** is an off-policy reinforcement learning algorithm that uses a neural network to approximate the Q-value function. This implementation includes the following features:
- **Epsilon-greedy exploration** with a decaying epsilon value for more efficient exploration and exploitation.
- **Experience replay buffer** to store agent experiences and sample random batches for training, which improves learning efficiency.
- **Target network** to stabilize the learning process by using a target network to compute Q-values for the Bellman equation.

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