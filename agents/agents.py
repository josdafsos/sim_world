""" Module containing agents implementations """

import random

from agents.base_agents import Agent, DQNBaseClass, MemoryFrameStack, EvolBaseClass
import creatures
import math


class RandomCow(Agent):

    def __init__(self):
        _, self.action_space = creatures.Cow.get_observation_action_spaces()

    def predict(self, obs) -> tuple[tuple[int, int], int]:

        #action = ((random.randint(-1, 1), random.randint(-1, 1)), random.randint(0, 2))
        action = random.randint(0, self.action_space - 1)
        return action

    def learn(self, old_obs, new_obs, action) -> None:
        """ It never learns anything"""
        pass


class DQNCow(DQNBaseClass):

    AGENT_NAME = "dqn_cow"

    def __init__(self,
                 agent_version: str = "new_agent",
                 verbose: int = 0,
                 epsilon: float | tuple[float, float, int] = 0.3,
                 learning_enabled: bool = True):

        super().__init__(agent_version,
                         verbose,
                         creatures.Cow,
                         epsilon=epsilon,
                         gradient_steps=-1,
                         learning_enabled=learning_enabled)

    def _get_is_done(self, new_obs, metadata={}):
        return metadata["species_cnt"] < 1  # creature is dead

    def _compute_reward(self, old_obs, new_obs, action, metadata={}):
        extra_penalty = 0
        extra_bonus = 0
        # extra_bonus += 5 * int(new_obs[4] > 2)  # extra reward for staying together
        # extra_penalty += 1 * int(new_obs[4] < 0.2)  # small penalty for walking alone to avoid unnecessary splits,
        extra_penalty += 1 * int(metadata["species_cnt"] < 1)  # if zero creatures left, it is dead and extra penalty for it
        reward = metadata["species_cnt_change"] - extra_penalty + extra_bonus  # species_cnt_change value

        return reward


class DQNWolf(DQNBaseClass):

    AGENT_NAME = "dqn_wolf"

    def __init__(self,
                 agent_version: str = "new _agent",
                 verbose: int = 0,
                 learning_enabled: bool = True):

        super().__init__(agent_version,
                         verbose,
                         creatures.Wolf,
                         epsilon=0.1,
                         gradient_steps=-1,
                         learning_enabled=learning_enabled)

    def _get_is_done(self, new_obs):
        return new_obs[4] < 1e-5  # creature is dead

    def _compute_reward(self, old_obs, new_obs, action):
        extra_penalty = 0
        # extra_penalty = 0.01 * int(new_obs[4] == 1)  # small penalty for walking alone to avoid unnecessary splits
        reward = new_obs[5] - extra_penalty  # species_cnt_change value

        return reward


class DQNMemoryWolf(MemoryFrameStack, DQNBaseClass):

    AGENT_NAME = "dqn_memory_wolf"

    def __init__(self, agent_version: str = "new_agent", verbose: int = 0):
        MemoryFrameStack.__init__(self, memory_frame_stack_length=10)
        DQNBaseClass.__init__(self, agent_version,
                              verbose,
                              creatures.Wolf,
                              epsilon=0.20,
                              gradient_steps=-1)

    def _get_is_done(self, new_obs):
        return new_obs[4] < 1e-5  # creature is dead

    def _compute_reward(self, old_obs, new_obs, action):
        extra_penalty = 0
        # extra_penalty = 0.01 * int(new_obs[4] == 1)  # small penalty for walking alone to avoid unnecessary splits
        reward = new_obs[5] - extra_penalty  # species_cnt_change value
        return reward


class NeatCow(EvolBaseClass):

    def __init__(self, model):
        super().__init__(creature_cls_or_operation_space=creatures.Cow)
        self.model = model

    def _compute_reward(self, old_obs, new_obs, action, metadata={}):
        extra_penalty = 0
        extra_bonus = 0
        # extra_bonus += 5 * int(new_obs[4] > 2)  # extra reward for staying together
        # extra_penalty += 1 * int(new_obs[4] < 0.2)  # small penalty for walking alone to avoid unnecessary splits,
        extra_penalty += 1 * int(metadata["species_cnt"] < 1)  # if zero creatures left, it is dead and extra penalty for it
        reward = metadata["species_cnt_change"] - extra_penalty + extra_bonus  # species_cnt_change value

        return reward

    def predict(self, obs) -> int:
        action = self.model.activate(obs)[0]
        # single output continuous action is mapped to integer value
        return round(self.action_space * (math.atan(action) + math.pi/2) / math.pi) - 1

