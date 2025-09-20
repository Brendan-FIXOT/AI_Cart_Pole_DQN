import gymnasium as gym

class CartPoleEnv:
    def __init__(self, render_mode=None):
        self.env = gym.make("CartPole-v1", render_mode=render_mode)
        self.state, _ = self.env.reset()  # Initialisation de l'état
        self.done = False

    def reset(self):
        self.state, _ = self.env.reset()
        self.done = False
        return self.state

    def step(self, action):
        if not self.done:
            self.state, reward, finished, truncated, _ = self.env.step(action)
            self.done = bool(finished or truncated)
            return self.state, reward, self.done
        else:
            return None, 0, True

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()