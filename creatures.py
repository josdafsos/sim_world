import random
import warnings

import pygame
import numpy as np

import world_properties
from base_creatures import Creature
import base_creatures
from creature_actions import *
import vegetation


class Cow(Creature):
    # For observations, instead of giving edible or not, provide the plant type (how to deal with a list?). And probably
    # plant group also. That is more challenging, but animals could learn to distinguish various vegetation types

    TYPE = "cow"
    MAX_HP = 10.0
    MAX_MOVEMENT_POINTS = 10.0
    MAX_FOOD_SUPPLY = 10.0
    CONSUMABLE_FOOD_TYPES = (vegetation.Grass.GROUP,)  # "fruit", "meat", "rotten meat"
    MAX_SPECIES_CNT = 10  # species amount the tile cannot overreach this value
    CREATURE_ID = world_properties.COW_ID  # Zero 0 reserved for no creature. Used by agents to distinguish between different species.
    CREATURE_NORMALIZED_ID = np.tanh(CREATURE_ID)  # Normalization for NN training
    VISION_DISTANCE = 1  # defines how far the creature can see
    DAYS_TO_BODY_BE_ROTTEN = 15  # body disappears after this number of days
    CHANCE_OF_BIRTH = 0.01  # 0.005  # on new day if the creature is in good condition there is chance to increase
    # population per each existing creature in stack
    IS_AFFECTING_ROADS = True  # defines if the creature can make a road by frequent walking on a tile
    SINGLE_CREATURE_STRENGTH = 1.0  # defines how much damage a creature per unit does during attack

    AVAILABLE_ACTIONS = (Eat,
                         Sleep,
                         Move,
                         Split, )
    OBSERVATION_SPACE = (42,)  # 106 for radius 2
    ACTION_SPACE = 'auto'

    def __init__(self, agent,
                 self_tile=None,
                 texture: str | tuple[int, int, int] = "cow_t.png",  #(255, 50, 50),
                 verbose: int = 0,
                 creature_to_copy=None,
                 *args, **kwargs):

        super().__init__(agent,
                         self_tile,
                         texture,
                         verbose,
                         creature_to_copy)

        # setting walking mode of an amphibious animal
        self.get_movement_difficulty = base_creatures.get_movement_difficulty_walk_swim

    def get_current_obs(self):
        # first set of own states, next state of the tiles in vision range
        # all data is normalized to [-1, 1] range
        obs = [
            self.CREATURE_NORMALIZED_ID,  # will be useful when meeting other creatures
            self.current_hp / self.MAX_HP,
            self.current_food / self.MAX_FOOD_SUPPLY,
            self.movement_points / self.MAX_MOVEMENT_POINTS,
            self.species_cnt / self.MAX_SPECIES_CNT,
            self.species_cnt_change,
        ]
        nearby_tiles = self.tile.get_surrounding_tile(self.VISION_DISTANCE)
        for tile in nearby_tiles:
            has_vegetation, _ = tile.has_vegetation_group(self.CONSUMABLE_FOOD_TYPES)
            creature = self.world.get_creature_on_tile(tile)
            if creature is None:
                creature_id = 0
                creature_cnt = 0
            else:
                creature_id = creature.CREATURE_NORMALIZED_ID
                creature_cnt = creature.species_cnt / creature.MAX_SPECIES_CNT

            sub_obs = [
                int(has_vegetation),
                self.get_movement_difficulty(self.tile, tile),
                creature_id,  # Other creature id if it is presented at the tile
                creature_cnt,  # Other creature count if it is presented at the tile
            ]
            obs.extend(sub_obs)

        self._obs_metadata_dict = {
            "species_cnt_change": self.species_cnt_change,
            "species_cnt": self.species_cnt,
        }

        obs = np.array(obs)
        return obs


class Wolf(Creature):
    TYPE = "wolf"
    MAX_HP: float = 10.0
    MAX_MOVEMENT_POINTS: float = 12.0
    MAX_FOOD_SUPPLY: float = 15.0
    CONSUMABLE_FOOD_TYPES = (world_properties.MEAT,)  # "fruit", "meat", "rotten meat"
    MAX_SPECIES_CNT: int = 10  # species amount the tile cannot overreach this value
    CREATURE_ID: int = world_properties.WOLF_ID  # Zero 0 reserved for no creature. Used by agents to distinguish between different species.
    CREATURE_NORMALIZED_ID: float = np.tanh(CREATURE_ID)  # Normalization for NN training
    VISION_DISTANCE: int = 1  # defines how far the creature can see
    CHANCE_OF_BIRTH = 0.020  # on new day if the creature is in good condition there is chance to increase
    # population per each existing creature in stack
    SINGLE_CREATURE_STRENGTH: float = 4.0  # defines how much damage a creature per unit does during attack

    AVAILABLE_ACTIONS = (EatAround,
                         Sleep,
                         Move,
                         Split, )
    OBSERVATION_SPACE: tuple[int,] = (42,)  # 106 for radius 2
    ACTION_SPACE = 'auto'

    def __init__(self, agent,
                 self_tile=None,
                 texture: str | tuple[int, int, int] = "wolf_t.png",  #(255, 50, 50),
                 verbose: int = 0,
                 creature_to_copy=None,
                 *args, **kwargs):

        super().__init__(agent,
                         self_tile,
                         texture,
                         verbose,
                         creature_to_copy)

        # setting walking mode of an amphibious animal
        self.get_movement_difficulty = base_creatures.get_movement_difficulty_walk_swim

    def get_current_obs(self):
        # first set of own states, next state of the tiles in vision range
        # all data is normalized to [-1, 1] range
        obs = [
            self.CREATURE_NORMALIZED_ID,  # will be useful when meeting other creatures
            self.current_hp / self.MAX_HP,
            self.current_food / self.MAX_FOOD_SUPPLY,
            self.movement_points / self.MAX_MOVEMENT_POINTS,
            self.species_cnt / self.MAX_SPECIES_CNT,
            self.species_cnt_change,
        ]
        nearby_tiles = self.tile.get_surrounding_tile(self.VISION_DISTANCE)
        for tile in nearby_tiles:
            has_food = tile.has_meat()
            creature = self.world.get_creature_on_tile(tile)
            if creature is None:
                creature_id = 0
                creature_cnt = 0
            else:
                creature_id = creature.CREATURE_NORMALIZED_ID
                creature_cnt = creature.species_cnt / creature.MAX_SPECIES_CNT
            sub_obs = [
                int(has_food),
                self.get_movement_difficulty(self.tile, tile),
                creature_id,  # Other creature id if it is presented at the tile
                creature_cnt,  # Other creature count if it is presented at the tile
            ]
            obs.extend(sub_obs)

        obs = np.array(obs)
        return obs

