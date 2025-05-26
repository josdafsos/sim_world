import random
import os
import numpy as np
from pathlib import Path
from datetime import datetime

from stable_baselines3 import DQN
from stable_baselines3.common import logger
import gymnasium as gym
import torch

import creatures
from creatures import Creature
import utils


AGENT_SAVING_FREQUENCY = 20_000

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


class DQNBaseClass(Agent):
    AGENT_NAME = "Agent Base Class"

    def __init__(self, agent_version,
                 verbose,
                 action_space,
                 learning_rate=1e-5,
                 batch_size=32,
                 ):

        self.model = None
        self.learning_rate: float = learning_rate  # for the model
        self.batch_size: int = batch_size  # for model
        self.verbose: int = verbose  # if > 0 then outputs agent states and related info
        self.actions: list = []
        self.action_space = action_space

        self.sum_reward = 0  # total reward before model is saved
        self.sum_steps = 0  # total steps before model is saved

        self.epsilon: float = 0.1  # defines the probability of taking a non-greedy action
        self.gradient_steps = 1  # for NN training, can be re-defined in child class

        self._saving_counter: int = 0  # counter to save model after a certain number of learning steps

        self.model_path = os.path.join(utils.agents_folder, self.AGENT_NAME, agent_version)  # path to the model

    def _load_model(self):
        if self.verbose > 0:
            print("loading DQN model: ", self.model_path)
        custom_objects = {'learning_rate': self.learning_rate}
        self.model = DQN.load(self.model_path, custom_objects=custom_objects)  # vec_env,
        # self.model.set_env(vec_env)
        # self.model = DQN.load(os.path.join('sb_neural_networks', 'dqn', model_name), vec_env, custom_objects=custom_objects)
        # self.model.set_env(vec_env)

    def _save_model(self):
        if self.verbose > 0:
            if self.sum_steps != 0:
                average_reward = self.sum_reward / self.sum_steps
            else:
                average_reward = 0
                print("sum steps equals zero, average reward cannot be computed")
            self.sum_steps = 0
            self.sum_reward = 0
            print("saving model, " + self.AGENT_NAME + " ", datetime.now(), ". Average reward for saving iteration: ", average_reward)
        self.model.save(self.model_path)

    def _train_network(self):
        if self.model.replay_buffer.size() > self.model.batch_size:
            self.model.train(batch_size=self.model.batch_size, gradient_steps=self.gradient_steps)
            self._saving_counter += 1
            if self.verbose > 2:
                print("training ongoing")
            if self._saving_counter > AGENT_SAVING_FREQUENCY:
                self._save_model()
                self._saving_counter = 0

    def predict(self, obs) -> tuple[tuple[int, int], int]:
        if random.random() < self.epsilon:
            action_idx = random.randint(0, self.action_space-1)
        else:
            action_idx, _ = self.model.predict(obs, deterministic=True)
        x, y, action = self.actions[action_idx]
        return ((x, y), action)


class DQNCow(DQNBaseClass):

    AGENT_NAME = "dqn_cow"

    def __init__(self, agent_version: str = "new agent", verbose: int = 0):

        super().__init__(agent_version, verbose, creatures.Creature.ACTION_SPACE)
        self.epsilon = 0.025
        self.gradient_steps = -1

        if Path(self.model_path + ".zip").exists():
            self._load_model()
        else:
            self._get_new_model()

        self.model._logger = logger.configure(None)  # we don't need any logger, but without the configuration model crashes

        self.actions = [(x, y, z)
                        for x in [-1, 0, 1]
                        for y in [-1, 0, 1]
                        for z in [0, 1, 2]]

    def _get_new_model(self):
        env = EmptyEnvironment(obs_shape=creatures.Creature.OBSERVATION_SPACE, n_actions=self.action_space)
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

    def learn(self, old_obs, new_obs, action):
        extra_penalty = 0
        extra_bonus = 0
        extra_bonus += 5 * int(new_obs[4] > 2)  # extra reward for staying together
        # extra_penalty += 1 * int(new_obs[4] < 0.2)  # small penalty for walking alone to avoid unnecessary splits,
        extra_penalty += 1000 * int(new_obs[4] < 1e-5)  # if zero creatures left, it is dead and extra penalty for it
        if action[1] == 2:  # the creature just split, usually not a good approach for a cow
            extra_penalty += 100

        # note, species count is normalized, thus < 0.3 means less than 30% of the max number of species
        reward = 100*new_obs[5] - extra_penalty + extra_bonus  # species_cnt_change value
        self.sum_reward += reward
        self.sum_steps += 1

        done = new_obs[4] < 1e-5  # creature is dead
        info = [{}]  # info must be a list of dicts
        (xy, z) = action  # converting action back to index
        x, y = xy
        x += 1
        y += 1
        action_idx = x * 9 + y * 3 + z  # TODO convert properly to action number
        action_idx = np.array([action_idx])
        self.model.replay_buffer.add(old_obs, new_obs, action_idx, reward, done, info)

        self._train_network()

class DQNWolf(DQNBaseClass):

    AGENT_NAME = "dqn_wolf"

    def __init__(self, agent_version: str = "new agent", verbose: int = 0):

        super().__init__(agent_version, verbose, creatures.Creature.ACTION_SPACE)
        self.epsilon = 0.10
        self.gradient_steps = -1

        if Path(self.model_path + ".zip").exists():
            self._load_model()
        else:
            self._get_new_model()

        self.model._logger = logger.configure(None)  # we don't need any logger, but without the configuration model crashes

        self.actions = [(x, y, z)
                        for x in [-1, 0, 1]
                        for y in [-1, 0, 1]
                        for z in [0, 1, 2]]

    def _get_new_model(self):
        env = EmptyEnvironment(obs_shape=creatures.Creature.OBSERVATION_SPACE, n_actions=self.action_space)
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

    def learn(self, old_obs, new_obs, action):
        extra_penalty = 0.01 * int(new_obs[4] == 1)  # small penalty for walking alone to avoid unnecessary splits
        reward = new_obs[5] - extra_penalty  # species_cnt_change value
        self.sum_reward += reward
        self.sum_steps += 1

        done = new_obs[5] <= 0  # creature is dead
        info = [{}]  # info must be a list of dicts
        (xy, z) = action  # converting action back to index
        x, y = xy
        x += 1
        y += 1
        action_idx = x * 9 + y * 3 + z  # TODO convert properly to action number
        action_idx = np.array([action_idx])
        self.model.replay_buffer.add(old_obs, new_obs, action_idx, reward, done, info)

        self._train_network()

