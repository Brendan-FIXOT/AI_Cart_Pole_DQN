# CartPole RL Agents (DQN, A2C, PPO)

This repository contains implementations of reinforcement learning algorithms applied to the **CartPole** problem from OpenAI Gym.  
It includes **Deep Q-Network (DQN)**, **Advantage Actor-Critic (A2C)**, and **Proximal Policy Optimization (PPO)**.  
The project demonstrates how different RL approaches can solve the same environment.

# CartPole RL Agents

![CartPole PPO Demo](assets/cartpolePPOv2.gif)


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

The **CartPole** problem involves balancing a pole on top of a moving cart.  
The agent must apply forces to the cart in order to keep the pole upright for as long as possible.  
Different reinforcement learning methods are implemented:

- **DQN**: Off-policy algorithm using Q-learning with a neural network.
- **A2C**: On-policy actor-critic method using value estimation to reduce variance.
- **PPO**: On-policy algorithm improving stability with a clipped surrogate objective.

### Algorithms and Features

- **DQN**
  - Epsilon-greedy exploration with decaying epsilon.
  - Experience replay buffer to improve sample efficiency.
  - Target network to stabilize learning.
- **A2C**
  - Two networks: actor (policy) and critic (state value).
  - Advantage estimation for policy gradient updates.
- **PPO (basic version)**
  - Actor-Critic framework with clipped objective to prevent large policy updates.
  - Uses a rollout buffer for training.

### Hyperparameters (default values)

| Hyperparameter       | Value            | Description |
|----------------------|------------------|-------------|
| **Learning Rate**     | 0.001 (actor/critic may differ) | Optimizer step size. |
| **Hidden size**  | 128 | Number of neurons in hidden layers |
| **Batch Size (DQN)**  | 64               | Training batch size for replay buffer. |
| **Epsilon (DQN)**     | 0.05 (decaying)  | Exploration probability. |
| **Gamma**             | 0.99             | Discount factor for future rewards. |
| **Clip value (PPO)** | 0.2 | Controls how much the new policy is allowed to deviate from the old one (stability vs learning speed). |
| **Buffer Size**       | 10,000 (DQN), 512 (PPO rollout) | Experience memory size. |
| **Epochs**            | 1000             | Number of training episodes. |

### Performance Metrics
- **Total Reward** tracked during training and testing.

### Best Models
- **DQN**: Achieved an average reward of ~517.6 over 100 test episodes after 1000 episodes of training. Longer training (e.g. 3000 episodes) often led to worse performance due to instability.
- **A2C**: Stable performance close to the maximum reward (≈500).
- **PPO**: Stable performance close to the maximum reward (≈500) with rollout of 512, if we increase the size, the model learn not enought.

## Getting Started

### Prerequisites

Before running the code, ensure that you have Python installed. You will also need:

- Python 3.x
- PyTorch
- OpenAI Gym
- NumPy

### Installing Dependencies

To install the required dependencies:

```bash
pip install -r requirements.txt
