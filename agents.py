import random
import os
from pathlib import Path

import utils

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

    def __init__(self, agent_version: str = "new agent"):
        self.name = "dqn_cow"
        self.model_path = os.path.join(utils.agents_folder, self.name, agent_version)
        self.model = None
        if Path(self.model_path + ".zip").exists():
            self._load_model()
        else:
            self._get_new_model()



    def _get_new_model(self):
        pass

    def _load_model(self):
        pass