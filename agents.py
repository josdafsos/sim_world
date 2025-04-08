import random


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
        action = ((random.randint(-1, 1), random.randint(-1, 1)), random.randint(0, 1))
        return action

    def learn(self, old_obs, new_obs, action) -> None:
        """ It never learns anything"""
        pass

class DQNCow(Agent):

    def __init__(self):
        pass