import gymnasium as gym
import cv2
import numpy as np

class BreakoutEnv:
    def __init__(self, render_mode=None):
        self.env = gym.make("Breakout-v4", render_mode=render_mode)
        self.observation_space = (84, 84, 1)   # Image 84x84 en niveaux de gris
        self.action_space = self.env.action_space

    def preprocess(self, obs):
        obs_gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs_resized = cv2.resize(obs_gray, (84, 84), interpolation=cv2.INTER_AREA)
        obs_normalized = obs_resized / 255.0
        return np.expand_dims(obs_normalized, axis=-1).astype(np.float32)

    def reset(self):
        obs, info = self.env.reset()
        return self.preprocess(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.preprocess(obs), reward, terminated, truncated, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()