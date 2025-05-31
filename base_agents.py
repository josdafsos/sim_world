"""

This module contains base classes for agents implementation.

"""

import os
import random
from datetime import datetime
from pathlib import Path
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common import logger
import gymnasium as gym
import torch

import utils


class Agent:
    def predict(self, obs) -> tuple[tuple[int, int], int]:
        """
        Predicts action based on observation,
        action is in format: (tile relative number {(-1, -1), (1,0), (-1,1), etc.}, action number)
        """
        pass

    def learn(self, old_obs, new_obs, action) -> None:
        """ Implements learning of an agent. Unnecessary to implement if agent does not learn. """
        pass


class EmptyEnvironment(gym.Env):
    """ Supplementary class to create StableBaselines agents. """

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


class DQNBaseClass(Agent):
    AGENT_NAME = "DQNBaseClass"
    AGENT_SAVING_FREQUENCY = 20_000

    def __init__(self, agent_version,
                 verbose: int,
                 action_space,
                 observation_space,
                 learning_rate: float = 1e-5,
                 batch_size: int = 32,
                 epsilon: float = 0.1,
                 gradient_steps: int = 1,
                 ):

        self.model = None
        self.learning_rate: float = learning_rate  # for the model
        self.batch_size: int = batch_size  # for model
        self.verbose: int = verbose  # if > 0 then outputs agent states and related info
        self.actions: list = []
        self.action_space = action_space
        self.observation_space = observation_space

        self.sum_reward: float = 0  # total reward before model is saved
        self.sum_steps: int = 0  # total steps before model is saved

        self.epsilon: float = epsilon  # defines the probability of taking a non-greedy action
        self.gradient_steps: int = gradient_steps  # for NN training, can be re-defined in child class

        self._saving_counter: int = 0  # counter to save model after a certain number of learning steps

        self.model_path = os.path.join(utils.agents_folder, self.AGENT_NAME, agent_version)  # path to the model

        # defines the topology of the neural network
        self.policy_kwargs = dict(
            net_arch=[256, 256],  # hidden layers with VALUE neurons each
            # activation_fn=torch.nn.ReLU
            activation_fn=torch.nn.ELU
        )

        if Path(self.model_path + ".zip").exists():
            self._load_model()
        else:
            self._get_new_model()

        self.model._logger = logger.configure(None)  # we don't need any logger, but without the configuration model crashes

    def _load_model(self):
        if self.verbose > 0:
            print("loading DQN model: ", self.model_path)
        custom_objects = {'learning_rate': self.learning_rate}
        self.model = DQN.load(self.model_path, custom_objects=custom_objects)  # vec_env,
        # self.model.set_env(vec_env)
        # self.model = DQN.load(os.path.join('sb_neural_networks', 'dqn', model_name), vec_env, custom_objects=custom_objects)
        # self.model.set_env(vec_env)

    def _get_new_model(self):
        env = EmptyEnvironment(obs_shape=self.observation_space, n_actions=self.action_space)
        self.model = DQN("MlpPolicy",
                         env,
                         learning_rate=self.learning_rate,
                         policy_kwargs=self.policy_kwargs,
                         exploration_fraction=0.3,
                         exploration_initial_eps=0.07,
                         exploration_final_eps=0.07,
                         gradient_steps=self.gradient_steps,  # -1,  # suggested by Ming, default 1
                         batch_size=self.batch_size,
                         verbose=1, )

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
            if self._saving_counter > self.AGENT_SAVING_FREQUENCY:
                self._save_model()
                self._saving_counter = 0

    def _compute_reward(self, old_obs, new_obs, action) -> float:
        """ Computes a reward for a given state"""
        pass

    def _get_is_done(self, new_obs) -> bool:
        """ Returns True if an episode is finished for the agent. Returns False otherwise """
        pass

    def predict(self, obs) -> tuple[tuple[int, int], int]:
        if random.random() < self.epsilon:
            action_idx = random.randint(0, self.action_space-1)
        else:
            action_idx, _ = self.model.predict(obs, deterministic=True)
        # x, y, action = self._idx_to_action(action_idx) # self.actions[action_idx]
        return  action_idx # self._idx_to_action(action_idx)
        #return ((x, y), action)

    def learn(self, old_obs, new_obs, action):
        reward = self._compute_reward(old_obs, new_obs, action)

        self.sum_reward += reward
        self.sum_steps += 1

        done = self._get_is_done(new_obs)
        info = [{}]  # info must be a list of dicts

        action_idx = action   # self._action_to_idx(action)
        self.model.replay_buffer.add(old_obs, new_obs, action_idx, reward, done, info)
        self._train_network()




