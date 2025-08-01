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

    def learn(self, old_obs, new_obs, action, metadata) -> None:
        """ Implements learning of an _agent. Unnecessary to implement if _agent does not learn. """
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


class MemoryFrameStack:
    """
        Class implements stack of observation frames. Any class to inherit it must contain
        observation_space attribute. Note: value of observation_space attribute will be modified.
    """

    def __init__(self, memory_frame_stack_length: int, *args, **kwargs):
        """
        :param memory_frame_stack_length: {positive integer}, if more than 1, requires several observation frames
        to stack together
        :param args:
        :param kwargs:
        """

        # super().__init__(*args, **kwargs)
        assert not hasattr(self, 'observation_space'), "MemoryFrameStack class must be initializaed before observation space attribute"

        if memory_frame_stack_length < 1:
            raise ValueError("memory_frame_stack_length must be positive")
        else:
            self._memory_frame_stack_length = memory_frame_stack_length

    def get_memory_frame_stack_length(self):
        return self._memory_frame_stack_length


class Observable:
    """
        Class to implement observation space.
        Must be inherited by any agent that intended to have an observation space.
        The class must be inherited after all observation space modifying classes.
    """

    def __init__(self, observation_space):
        self.observation_space = np.array(observation_space)

        if isinstance(self, MemoryFrameStack):
            assert hasattr(self, '_memory_frame_stack_length'), "MemoryFrameStack must be inherited before any class containing Observable"
            self.observation_space = self.observation_space * self._memory_frame_stack_length


class DQNBaseClass(Agent, Observable):
    """ Base class for all DQN agents """

    AGENT_NAME = "DQNBaseClass"
    AGENT_SAVING_FREQUENCY = 1_000_000

    def __init__(self, agent_version,
                 verbose: int,
                 creature_cls_or_operation_space: tuple | type,
                 learning_rate: float = 1e-5,
                 batch_size: int = 32,
                 epsilon: float | tuple[float, float, int] = 0.1,
                 gradient_steps: int = 1,
                 policy_kwargs: dict = None,
                 learning_enabled: bool = True,
                 *args,
                 **kwargs
                 ):
        """
        :param agent_version:
        :param verbose: {0, 1, 2} if more than 0 adds status information into console,
        1 - base information, 2 - all information
        :param creature_cls_or_operation_space: class of a creature to be controlled by the agent. Use to automatically get
        observation and action spaces. Alternatively, tuple[int, int] can be given with manually set
        [observation space, action space] values
        :param learning_rate:
        :param batch_size:
        :param epsilon: float [0, 1] probability of taking a random action instead of using agent's predict function
        tuple(initial epsilon value, final epsilon value, step to reach final value),
        epsilon will be linearly reduced until it reaches the final value at which it will be kept.
        :param gradient_steps: {-1, positive integer} number of gradient steps for training the agent, -1 for max steps
        :param policy_kwargs: dictionary to define neural network topology. If None, default topology is used
        :param memory_frame_stack_length: {positive integer} number of observation frames stacked together
        :param learning_enabled - default True. Allows agent to learn. Set to False to disable learning and
        decrease computation time
        """

        # super().__init__(*args, **kwargs)

        self.model = None
        self.learning_rate: float = learning_rate  # for the model
        self.batch_size: int = batch_size  # for model
        self.verbose: int = verbose  # if > 0 then outputs _agent states and related info
        self.learning_enabled = learning_enabled

        if isinstance(creature_cls_or_operation_space, tuple):
            observation_space = creature_cls_or_operation_space[0]
            action_space = creature_cls_or_operation_space[1]
        else:
            observation_space, action_space = creature_cls_or_operation_space.get_observation_action_spaces()

        Observable.__init__(self, observation_space)
        self.action_space = action_space

        self.sum_reward: float = 0  # total reward before model is saved
        self.sum_steps: int = 0  # total steps before model is saved
        self._steps_done: int = 0  # counter on how many predictions has been made since agent's initialization

        if isinstance(epsilon, float):
            self.epsilon: float = epsilon  # defines the current probability of taking a non-greedy action
            self.initial_epsilon, self.final_epsilon, self.max_epsilon_steps = None, None, None
        else:
            self.initial_epsilon = epsilon[0]
            self.epsilon = self.initial_epsilon
            self.final_epsilon = epsilon[1]
            self.max_epsilon_steps = epsilon[2]

        self.gradient_steps: int = gradient_steps  # for NN training, can be re-defined in child class
        self._saving_counter: int = 0  # counter to save model after a certain number of learning steps

        self.model_path = os.path.join(utils.agents_folder, self.AGENT_NAME, agent_version)  # path to the model

        # defines the topology of the neural network
        if policy_kwargs is None:
            self._policy_kwargs = dict(
                net_arch=[256, 256],  # hidden layers with VALUE neurons each
                # activation_fn=torch.nn.ReLU
                activation_fn=torch.nn.ELU
            )
        else:
            self._policy_kwargs = policy_kwargs
        print(Path(self.model_path + ".zip"))
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
                         policy_kwargs=self._policy_kwargs,
                         exploration_fraction=0.3,
                         exploration_initial_eps=0.07,
                         exploration_final_eps=0.07,
                         gradient_steps=self.gradient_steps,  # -1,  # suggested by Ming, default 1
                         batch_size=self.batch_size,
                         verbose=1, )

        if self.verbose > 0:
            print("New model was generated for ", self.AGENT_NAME)
            print(self.model.policy)

    def _save_model(self):
        if self.verbose > 0:
            if self.sum_steps != 0:
                average_reward = self.sum_reward / self.sum_steps
            else:
                average_reward = 0
                print("sum steps equals zero, average reward cannot be computed")
            self.sum_steps = 0
            self.sum_reward = 0
            print("saving model, " + self.AGENT_NAME + " ", datetime.now(),
                  ". Average reward for saving iteration: ", average_reward,
                  " Epsilon: ", self.epsilon)
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

    def _compute_reward(self, old_obs, new_obs, action, metadata: dict = {}) -> float:
        """ Computes a reward for a given state"""
        pass

    def _get_is_done(self, new_obs, metadata: dict = {}) -> bool:
        """ Returns True if an episode is finished for the _agent. Returns False otherwise """
        pass

    def predict(self, obs) -> int:
        if random.random() < self.epsilon:
            action_idx = random.randint(0, self.action_space-1)
        else:
            action_idx, _ = self.model.predict(obs, deterministic=True)

        if self.max_epsilon_steps is not None:  # linearly decreasing epsilon if required
            self.epsilon = (self.final_epsilon +
                            (self.initial_epsilon - self.final_epsilon) *
                            (1 - self._steps_done / self.max_epsilon_steps))

            if self._steps_done >= self.max_epsilon_steps:
                self.max_epsilon_steps = None

        self._steps_done += 1
        return action_idx

    def learn(self, old_obs, new_obs, action, metadata: dict = {}):
        if not self.learning_enabled:
            return

        reward = self._compute_reward(old_obs, new_obs, action, metadata)

        self.sum_reward += reward
        self.sum_steps += 1

        done = self._get_is_done(new_obs, metadata)
        info = [{}]  # info must be a list of dicts

        action_idx = action   # self._action_to_idx(action)
        self.model.replay_buffer.add(old_obs, new_obs, action_idx, reward, done, info)
        self._train_network()




