""" Module containing agents implementations """

import random

from base_agents import Agent, DQNBaseClass, MemoryFrameStack
import creatures


class RandomCow(Agent):

    def __init__(self):
        pass

    def predict(self, obs) -> tuple[tuple[int, int], int]:

        action = ((random.randint(-1, 1), random.randint(-1, 1)), random.randint(0, 2))
        return action

    def learn(self, old_obs, new_obs, action) -> None:
        """ It never learns anything"""
        pass


class DQNCow(DQNBaseClass):

    AGENT_NAME = "dqn_cow"

    def __init__(self, agent_version: str = "new _agent", verbose: int = 0):

        super().__init__(agent_version,
                         verbose,
                         creatures.Creature.ACTION_SPACE,
                         creatures.Creature.OBSERVATION_SPACE,
                         epsilon=0.025,
                         gradient_steps=-1)

    def _get_is_done(self, new_obs):
        return new_obs[4] < 1e-5  # creature is dead

    def _compute_reward(self, old_obs, new_obs, action):
        extra_penalty = 0
        extra_bonus = 0
        extra_bonus += 5 * int(new_obs[4] > 2)  # extra reward for staying together
        # extra_penalty += 1 * int(new_obs[4] < 0.2)  # small penalty for walking alone to avoid unnecessary splits,
        extra_penalty += 1000 * int(new_obs[4] < 1e-5)  # if zero creatures left, it is dead and extra penalty for it
        # if action[1] == 2:  # the creature just split, usually not a good approach for a cow
        #     extra_penalty += 100

        # note, species count is normalized, thus < 0.3 means less than 30% of the max number of species
        reward = 100*new_obs[5] - extra_penalty + extra_bonus  # species_cnt_change value

        return reward


class DQNWolf(DQNBaseClass):

    AGENT_NAME = "dqn_wolf"

    def __init__(self, agent_version: str = "new _agent", verbose: int = 0):

        super().__init__(agent_version,
                         verbose,
                         creatures.Creature.ACTION_SPACE,
                         creatures.Creature.OBSERVATION_SPACE,
                         epsilon=0.1,
                         gradient_steps=-1)

    def _get_is_done(self, new_obs):
        return new_obs[4] < 1e-5  # creature is dead

    def _compute_reward(self, old_obs, new_obs, action):
        extra_penalty = 0
        # extra_penalty = 0.01 * int(new_obs[4] == 1)  # small penalty for walking alone to avoid unnecessary splits
        reward = new_obs[5] - extra_penalty  # species_cnt_change value
        return reward


class DQNMemoryWolf(DQNBaseClass, MemoryFrameStack):

    AGENT_NAME = "dqn_wolf"

    def __init__(self, agent_version: str = "new _agent", verbose: int = 0):

        MemoryFrameStack.__init__(self, memory_frame_stack_length=10)
        DQNBaseClass.__init__(self, agent_version,
                              verbose,
                              creatures.Creature.ACTION_SPACE,
                              creatures.Creature.OBSERVATION_SPACE,
                              epsilon=0.1,
                              gradient_steps=-1)

    def _get_is_done(self, new_obs):
        return new_obs[4] < 1e-5  # creature is dead

    def _compute_reward(self, old_obs, new_obs, action):
        extra_penalty = 0
        # extra_penalty = 0.01 * int(new_obs[4] == 1)  # small penalty for walking alone to avoid unnecessary splits
        reward = new_obs[5] - extra_penalty  # species_cnt_change value
        return reward
