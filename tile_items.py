import warnings
import random

from creatures import Creature
from world_properties import WorldProperties

"""
Module to implement items belonging to tiles, such as fire, water wells, and other 
entities and events that have a strict belonging to a particular tile. 
"""


class TileItem:
    HAS_PREPARE_STEP: bool = False  # for all items with such feature True,
    # prepare step calculations will be made on every step
    HAS_STEP: bool = False  # for all items with such feature True,
    # step calculations will be made on every step
    HAS_PASSIVE: bool = False  # for all items with the feature a passive triggering will be enabled by certain events (not specified which events exactly yet)
    DRAWING_PRIORITY: int = 1  # the higher the value, the later item will be drawn (i.e. will be less obscured by other items)
    TYPE: str = "BASE ITEM"  # There can be only one instance of particular TYPE on every tile

    def __init__(self, tile):
        """

        :param tile: Tile on which the item is initialized
        """
        super().__init__()
        self.tile = tile

    def prepare_step(self) -> None:
        """
        if self.HAS_PREPARE_STEP == True, then this function will be called on each terrain step
        :return:
        """
        pass

    def step(self) -> None:
        """
        if self.HAS_STEP == True, then this function will be called on each terrain step
        """
        pass

    def draw(self, screen, pos) -> None:
        """
        This function is called on each world rendering step
        """


class DeadBody(TileItem):
    HAS_STEP = True  # dead bodies are erase on each step
    TYPE = "dead body"

    def __init__(self, tile, bodies_cnt: int, creature: Creature):
        super().__init__(tile=tile)
        self.dead_cnt: int = bodies_cnt  # number of dead bodies on the tile
        self.dead_creature: Creature = creature  # temporary variable, will be used for drawing mostly

    def add_dead_creature(self, bodies_cnt: int, creature: Creature):
        """
        Adds a new dead body to the tile, that will be considered as meet
        :param bodies_cnt: number of new bodies
        :param creature: Creature to be used for drawing dead body
        :return:
        """
        self.dead_cnt += bodies_cnt
        self.dead_creature = creature

    def erase_dead_creature(self, bodies_cnt: int):
        """
        Reduces the number of bodies on the tile by a given amount.
        :param bodies_cnt: positive value of number of the bodies to be erased
        :return:
        """
        self.dead_cnt -= bodies_cnt
        if self.dead_cnt < 1:
            self.tile.delete_item(self)

    def step(self) -> None:
        if random.random() < WorldProperties.body_disappear_chance:
            bodies_to_erase = random.randint(1, self.dead_cnt // 2 + 1)  # if only one body is deleted the process takes too much time
            self.erase_dead_creature(bodies_to_erase)  # one body is erased at a time

    def draw(self, screen, pos) -> None:
        self.dead_creature.draw(screen, pos)


