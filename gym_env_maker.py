import torch
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt

class PortfolioOptimization(gym.Env):
    def __init__(self, data, n_steps=60, budget=10**10):
        super(PortfolioOptimization, self).__init__()
        self.data = data
        self.n_steps = n_steps
        self.current_step = 0
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(205,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(200,), dtype=np.float32)
        self.initial_budget = budget
        self.current_budget = budget

    def step(self, action):
        action = action / np.sum(action)
        reward = self.calculate_reward(action, self.data[self.current_step])
        self.budget_change(action, self.data[self.current_step])
        # if self.current_step % 5 == 0:
        #     print("{}th step's reward is {}".format(self.current_step, reward))
        
        info = {'current_budget': self.current_budget}

        self.current_step += 1
        if self.current_step >= len(self.data): 
            done = True
            self.current_step = 0
        else:
            done = False

        next_state = self.data[self.current_step]
        
        return next_state, reward, done, False, info

    def reset(self, **kwargs):
        seed = kwargs.get('seed', None)
        if seed is not None:
            np.random.seed(seed)
        
        self.current_step = np.random.randint(len(self.data) - self.n_steps)
        observation = self.data[self.current_step]
        self.current_budget = self.initial_budget

        if not self.observation_space.contains(observation):
            raise ValueError("Observation is not within the observation space.")

        return observation, {}
    
    def budget_change(self, action, state):
        budget_per_company = action * self.current_budget
        self.current_budget += np.dot(budget_per_company, state[4:-1])
    
    def calculate_reward(self, action, state):
        exp_return = action @ state[4:-1] 
        rf_rate = state[-1] / 365
        std = np.std(state[4:-1])
        if std == 0:
            std = 1e-4  # Larger epsilon for more stability
        sharpe_ratio = (exp_return - rf_rate) / std
        return np.clip(sharpe_ratio, -10, 10)


class CustomCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.budgets = []
        self.global_step = 0

    def init_callback(self, model):
        super().init_callback(model)
        num_envs = self.training_env.num_envs
        self.budgets = [[] for _ in range(num_envs)]

    def _on_step(self) -> bool:
        # Tracking budgets
        infos = self.locals['infos']
        for i, info in enumerate(infos):
            if 'current_budget' in info:
                self.budgets[i].append(info['current_budget'])

        # Linear decay learning_rate
        self.global_step += 1
        self.model.learning_rate *= 1-self.global_step/self.num_timesteps

        return True

    def _on_training_end(self) -> None:
        for env_idx, budget_log in enumerate(self.budgets):
            print("Budget log for environment{} : {}".format(env_idx, budget_log))

    def plot_best_budget_changes(self):
        best_env_idx = np.argmax([budget_log[-1] for budget_log in self.budgets])
        best_budget_log = self.budgets[best_env_idx]
        plt.figure(figsize=(10, 6))
        plt.plot(best_budget_log, label=f'Best Environment {best_env_idx}')
        plt.xlabel('Timesteps')
        plt.ylabel('Current Budget')
        plt.title('Best Environment Budget Changes Over Time')
        plt.legend()
        plt.show()

# Register the custom environment to gym
register(
    id='PO-v0',
    entry_point=PortfolioOptimization,
    kwargs={'data': None, 'n_steps': 60, 'budget': 10**10}
)