import random
import os
import numpy as np
from pathlib import Path

from stable_baselines3 import DQN
from stable_baselines3.common import logger
import gymnasium as gym
import torch

import creatures
from creatures import Creature
import utils


class EmptyEnvironment(gym.Env):
    def __init__(self, obs_shape=(4,), n_actions=2):
        super().__init__()

        # Define observation and action spaces
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)
        self.action_space = gym.spaces.Discrete(n_actions)

    def reset(self):
        # Return dummy observation
        return np.zeros(self.observation_space.shape, dtype=np.float32)

    def step(self, action):
        # Return dummy transition
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        reward = 0.0
        done = True
        info = {}
        return obs, reward, done, info

class Agent:
    def predict(self, obs) -> tuple[tuple[int, int], int]:
        """
        Predicts action based on observation,
        action is in format: (tile relative number {(-1, -1), (1,0), (-1,1), etc.}, action number)
        """
        pass

    def learn(self, old_obs, new_obs, action) -> None:
        """ Implements learning of an agent """
        pass


class RandomCow(Agent):

    def __init__(self):
        pass

    def predict(self, obs) -> tuple[tuple[int, int], int]:
        action = ((random.randint(-1, 1), random.randint(-1, 1)), random.randint(0, 2))
        return action

    def learn(self, old_obs, new_obs, action) -> None:
        """ It never learns anything"""
        pass


class DQNCow(Agent):

    def __init__(self, agent_version: str = "new agent", verbose: int=0):
        self.name = "dqn_cow"
        self.model_path = os.path.join(utils.agents_folder, self.name, agent_version)
        self.model = None
        self.learning_rate = 1e-5
        self.batch_size = 32
        self.verbose = verbose

        if Path(self.model_path + ".zip").exists():
            self._load_model()
        else:
            self._get_new_model()

        self.model._logger = logger.configure(None)  # we don't need any logger, but without the configuration model crashes

        self._saving_counter = 0
        self.actions = [(x, y, z)
                        for x in [-1, 0, 1]
                        for y in [-1, 0, 1]
                        for z in [0, 1, 2]]

    def _get_new_model(self):
        env = EmptyEnvironment(obs_shape=creatures.Creature.OBSERVATION_SPACE, n_actions=creatures.Creature.ACTION_SPACE)
        policy_kwargs = dict(
            net_arch=[256, 256],  # hidden layers with VALUE neurons each
            # activation_fn=torch.nn.ReLU
            activation_fn=torch.nn.ELU
        )

        self.model = DQN("MlpPolicy",
                         env,
                         learning_rate=self.learning_rate,
                         policy_kwargs=policy_kwargs,
                         exploration_fraction=0.3,
                         exploration_initial_eps=0.07,
                         exploration_final_eps=0.07,
                         gradient_steps=1,  # -1,  # suggested by Ming, default 1
                         batch_size=self.batch_size,
                         verbose=1, )

    def _load_model(self):
        if self.verbose > 0:
            print("loading DQN model")
        custom_objects = {'learning_rate': self.learning_rate}
        self.model = DQN.load(self.model_path, custom_objects=custom_objects)  # vec_env,
        # self.model.set_env(vec_env)
        # self.model = DQN.load(os.path.join('sb_neural_networks', 'dqn', model_name), vec_env, custom_objects=custom_objects)
        # self.model.set_env(vec_env)

    def _save_model(self):
        if self.verbose > 0:
            print("saving model")
        self.model.save(self.model_path)

    def learn(self, old_obs, new_obs, action):

        extra_penalty = 0.01 * new_obs[5]  # small penalty for walking alone to avoid unnecessary splits
        reward = new_obs[5] - extra_penalty  # species_cnt_change value
        done = new_obs[5] <= 0  # creature is dead
        info = [{}]  # info must be a list of dicts
        (xy, z) = action  # converting action back to index
        x, y = xy
        x += 1
        y += 1
        action_idx = x * 9 + y * 3 + z  # TODO convert properly to action number
        action_idx = np.array([action_idx])
        self.model.replay_buffer.add(old_obs, new_obs, action_idx, reward, done, info)

        if self.model.replay_buffer.size() > self.model.batch_size:
            self.model.train(batch_size=self.model.batch_size, gradient_steps=1)
            self._saving_counter += 1
            if self.verbose > 1:
                print("training ongoing")
            if self._saving_counter > 100_000:
                self._save_model()
                self._saving_counter = 0

    def predict(self, obs) -> tuple[tuple[int, int], int]:
        action_idx, _ = self.model.predict(obs, deterministic=False)
        x, y, action = self.actions[action_idx]
        return ((x, y), action)
