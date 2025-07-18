"""

This module contains base classes and static methods for creature implementation.

"""

import numpy as np
import pygame
import random
import warnings
from collections import deque

from base_agents import MemoryFrameStack
from graphics import graphics


def get_movement_difficulty_walk_swim(tile_initial, tile_to_move) -> float:
    """ Computes movement required for an amphibious animal to move"""

    movement_difficulty = 2.0
    if tile_initial.water.relative_height > 1e-5:
        movement_difficulty += 2.0
    elif tile_initial.water.moisture_level > tile_initial.water.MOISTURE_LEVEL_TO_RISE_HEIGHT * 0.3:
        movement_difficulty += 1.0
    if tile_to_move.water.relative_height > 1e-5:
        movement_difficulty += 2.0
    elif tile_to_move.water.moisture_level > tile_initial.water.MOISTURE_LEVEL_TO_RISE_HEIGHT * 0.3:
        movement_difficulty += 1.0
    if tile_initial.height_level - tile_to_move.height_level > 0:
        movement_difficulty -= 0.5
    else:
        movement_difficulty += 1.0

    return movement_difficulty


class Creature:
    # TODO for observations, feed not edible or not, but the plant time. And probably plant group also.
    # That is more challenging, but animals could learn to distinguish various vegetation types

    """ Base class for any creature """

    NAME: str = "default creature"
    MAX_HP: float = 10.0
    MAX_MOVEMENT_POINTS: float = 10.0
    MAX_FOOD_SUPPLY: float = 10.0
    CONSUMABLE_FOOD_TYPES: tuple[str, ...] = ("grass",)  # "fruit", "meat", "rotten meat"
    MAX_SPECIES_CNT: int = 10  # species amount the tile cannot overreach this value
    CREATURE_ID: int = None  # Zero 0 reserved for no creature. Used by agents to distinguish between different species.
    CREATURE_NORMALIZED_ID: float = None  # must be defined as follow: np.tanh(CREATURE_ID) Normalization for NN training
    VISION_DISTANCE: int = 1  # defines how far the creature can see
    DAYS_TO_BODY_BE_ROTTEN: int = 15  # body disappears after this number of days
    CHANCE_OF_BIRTH: float = 0.01  # 0.005  # on new day if the creature is in good condition there is chance to increase
    # population per each existing creature in stack
    MASS: float = 1.0  # mass of a single creature (kinda in kg), used for interactions
    IS_AFFECTING_ROADS: bool = True  # defines if the creature can make a road by frequent walking on a tile
    SINGLE_CREATURE_STRENGTH: float = 1.0  # defines how much damage a creature per unit does during attack

    OBSERVATION_SPACE: tuple[int,] = (42,)  # 106 for radius 2
    ACTION_SPACE: int = 27  # [{-1, 0, 1}, {-1, 0, 1}, {0, 1, 2}

    # activity that can be done towards selected tile,
    # options: move (also unify, reproduce, attack (if hostile), move onto the current tile = sleep); eat;
    # attack/split (if tile is empty, half of species go to the tile. If own tile is selected,
    # splits part of the species into random nearby location if free.
    # If the selected tile is occupied, the occupying creature is attacked even against friendly)
    AVAILABLE_ACTIONS: tuple[str, ...] = (
        "move", "eat", "split/attack")  # , "split")  # split/attack is not implemented yet

    def __init__(self, agent,
                 self_tile=None,
                 texture: str | tuple[int, int, int] = (255, 50, 50),
                 verbose: int = 0,
                 creature_to_copy=None):
        """
        Base class for any creautre

        :param agent: any _agent of class Agent to control the creature
        :param self_tile: tile on which creature is spawned. Can be None
        :param texture: picture or a color tuple
        :param verbose: 0 - no console prints, 1 - to print states of the creature
        :param creature_to_copy: a creature can be copied, does not override new _agent, texture or a tile
        """
        if creature_to_copy is None:
            self.current_hp: float = self.MAX_HP
            self.movement_points: float = self.MAX_MOVEMENT_POINTS
            self.current_food: float = 0.85 * self.MAX_FOOD_SUPPLY  # opposite of hunger value
            self.species_cnt: int = 5  # current number of species at the tile
            self.observation = None
            self.days_since_death: int = 0  # if positive sets the creature to the dead state. First day is the day of death
            self.species_cnt_change: int = 0  # indicates change in size of the stack for birth/death. Reset on new action.
            self.get_movement_difficulty = None  # function to compute movement difficulty accepting (tile1, tile2) as args
            # Does not indicate split
        else:
            self.current_hp: float = creature_to_copy.current_hp
            self.movement_points: float = creature_to_copy.movement_points
            self.current_food: float = creature_to_copy.current_food
            self.species_cnt: int = creature_to_copy.species_cnt
            self.observation = creature_to_copy.observation
            self.days_since_death: int = creature_to_copy.days_since_death
            self.species_cnt_change: int = 0
            self.get_movement_difficulty = creature_to_copy.get_movement_difficulty

        self._memory_frame_stack: int = 1  # defines how many observation frame will be stacked
        self._observation_buffer = None  # for observations frame stacking
        # together and given to agent
        self._agent = None  # declaring _agent attribute, actual initialization is inside the setter function
        self.set_agent(agent)
        self.tile = self_tile
        self.verbose = verbose
        self.world = None

        if isinstance(texture, str):
            #self.texture = pygame.image.load("textures//" + texture).convert_alpha()
            self.texture = graphics.get_texture(texture)
        else:
            self.texture = texture  # can be texture or a color of a square
        self.scaled_texture = None

        if self_tile is not None:
            self.set_tile(self_tile)

    def set_tile(self, tile):
        """ Sets tile, matches world to the tile and updates observations """
        self.tile = tile
        self.world = self.tile.world
        #self.observation = self.get_current_obs()
        self._update_obs()
        self.on_rescale()  # if texture exists scaling it

    def set_agent(self, agent):
        self._agent = agent
        if agent is not None:
            if isinstance(agent, MemoryFrameStack):  # implement frame stack memory
                self._memory_frame_stack = agent.get_memory_frame_stack_length()
                self._observation_buffer = deque(maxlen=self._memory_frame_stack)
                for _ in range(self._memory_frame_stack):
                    self._observation_buffer.append(np.zeros((self.OBSERVATION_SPACE[0], )))

    def new_day(self) -> bool:
        """
        Actions executed at the beginning of a new day
        :return True if the body is rotten and entity must be deleted, otherwise returns False
        """
        # self.movement_points = min(self.movement_points + 3, self.MAX_MOVEMENT_POINTS)
        if self.days_since_death > 0:
            self.days_since_death += 1
        if self.days_since_death > self.DAYS_TO_BODY_BE_ROTTEN:
            if self.verbose > 0:
                print("body is deleted")
            return True

        newborn_creatures = 0
        for i in range(self.species_cnt):
            if self.MAX_SPECIES_CNT <= self.species_cnt:
                break
            if (self.species_cnt > 1
                    and self.current_hp > 0.9 * self.MAX_HP
                    and self.current_food > 0.75 * self.MAX_FOOD_SUPPLY
                    and random.random() < self.CHANCE_OF_BIRTH):
                newborn_creatures += 1
        self.species_cnt += newborn_creatures
        self.species_cnt_change += newborn_creatures

        return False

    def make_actions(self):
        """
        Forces Creature to make actions while it has movement points or
        unless an action blocking further actions is made. Updates observations after an action is taken and
        feeds data into agent.learn()
        """

        if self.verbose > 0:
            print(f"creature {self} acts")
        if self.species_cnt < 1:
            warnings.warn("a dead creature is trying to make a move")
            self.apply_damage(1000.0)
            return

        if self.days_since_death > 0:
            # TODO implement activities happening with the body
            return

        if self._agent is None:
            warnings.warn("a creature with no _agent was requested to make an action")
            return

        has_done_action = False
        is_final_action = False

        while self.species_cnt > 0 and self.days_since_death < 1:
            # action tuple (tile relative number {(-1, -1), (1,0), (-1,1), etc}, action number)
            action = self._idx_to_action(self._agent.predict(self.observation))
            if self.verbose > 0:
                print(f"action: {action}, ({self.AVAILABLE_ACTIONS[action[1]]})")
            if self.AVAILABLE_ACTIONS[action[1]] == "move":
                if action[0] == (0, 0):
                    self._sleep(has_done_action)
                    is_final_action = True
                else:
                    self._move(action[0])
            elif self.AVAILABLE_ACTIONS[action[1]] == "eat":
                self._eat(action[0])
            elif self.AVAILABLE_ACTIONS[action[1]] == "split/attack":
                self._split_or_attack(action[0])
            else:
                warnings.warn(f"Warning, unknown action is called, action index: {action[1]}")
                break

            old_obs = self.observation
            self._update_obs()
            new_obs = self.observation
            self._agent.learn(old_obs, new_obs, self._action_to_idx(action))
            self.species_cnt_change = 0  # restoring the indication of death/birth
            has_done_action = True
            if self.species_cnt < 1 or self.days_since_death > 0:
                warnings.warn("dead creature was in the middle of making an action, interrupting...")
                self.apply_damage(1000)
                break

            if self.movement_points < 1e-4 or is_final_action:
                break

    def on_rescale(self):
        """ This function must be called if the width of Tile has changed"""
        # NOTE: this function could somehow be moved to graphics
        if not isinstance(self.texture, tuple) and self.world is not None:
            width = self.world.tile_width
            self.scaled_texture = pygame.transform.scale(self.texture,
                                                         (1.00 * width, 1.00 * width))  # previously 0.5*...

    def draw(self, screen, pos):  # , width=None, height_scale=None, height_pos=None):
        width = self.world.tile_width
        height_pos = [pos[0] + self.tile.height_level * self.world.height_direction[0] * self.world.height_scale,
                      pos[1] + self.tile.height_level * self.world.height_direction[1] * self.world.height_scale]

        if isinstance(self.texture, tuple):  # color instead of texture
            if self.days_since_death > 0:
                # drawing a dead body
                pygame.draw.rect(screen, (30, 30, 30), (height_pos[0], height_pos[1], 0.5 * width, 0.5 * width))
            else:
                pygame.draw.rect(screen, self.texture, (height_pos[0] + 0.05 * width,
                                                        height_pos[1] + 0.05 * width,  # 0.45*width
                                                        1.00 * width,
                                                        1.80 * width))
        else:  # in case a texture is used
            if self.days_since_death > 0:
                pygame.draw.rect(screen, (30, 30, 30), (height_pos[0], height_pos[1], 0.5 * width, 0.5 * width))
            else:
                self.world.screen.blit(self.scaled_texture,
                                       (height_pos[0] - 0.10 * width, height_pos[1] - 0.10 * width))

        if self.days_since_death < 1:

            if self.MAX_HP - self.current_hp > 1e-4:  # drawing hp bar if it is not maximum
                x = height_pos[0] + 0.05 * width
                y = height_pos[1] + 0.8 * width
                length = 0.6 * width
                hp_ratio = max(0, self.current_hp / self.MAX_HP)
                pygame.draw.line(screen,
                                 (0, 0, 0),
                                 (x, y),
                                 (x + length, y),
                                 6)
                pygame.draw.line(screen,
                                 (255, 0, 0),
                                 (x, y),
                                 (x + length * hp_ratio, y),
                                 4)

            text_surface = self.world.font.render(str(self.species_cnt), True, (255, 255, 255))
            self.world.screen.blit(text_surface, (height_pos[0] + 0.6 * width, height_pos[1] + 0.45 * width))

    def apply_damage(self, damage: float):
        self.current_hp -= damage
        if self.current_hp < 0.001:  # and self.days_since_death == 0:
            if damage < 1:
                if random.random() < damage:
                    dead_cnt = 1
                else:
                    dead_cnt = 0
            else:
                dead_cnt = random.randint(1, round(damage))
            self.species_cnt_change = -dead_cnt
            self.species_cnt -= dead_cnt
            if dead_cnt > 0:
                creature_copy = self.__class__(agent=None,
                                               self_tile=self.tile,
                                               texture=self.texture,
                                               creature_to_copy=self)
                creature_copy.days_since_death = 1
                self.tile.add_dead_creature(dead_cnt, creature_copy)  # self)
            if self.species_cnt < 1:
                self.species_cnt = 0
                self.world.remove_creature(self)
                self.days_since_death = 1
                if self.verbose > 0:
                    print(f"the creature died {self}")

    def apply_heal(self, hp_restored: float):
        self.current_hp = min(self.MAX_HP, self.current_hp + hp_restored)

    def consume_food(self, food_consumed, enable_heal=False):
        self.current_food -= food_consumed
        if self.current_food < 0:
            self.apply_damage(abs(self.current_food))
            self.current_food = 0.0
        elif enable_heal:
            self.apply_heal(food_consumed)

    def _attack(self, other_creature):
        own_attack = self.species_cnt * self.SINGLE_CREATURE_STRENGTH
        other_attack = other_creature.species_cnt * other_creature.SINGLE_CREATURE_STRENGTH
        min_damage = 0.1  # for some reason an attack can deal a negative damage, idk why
        own_attack = max(min_damage, own_attack)
        other_attack = max(min_damage, other_attack)

        self.apply_damage(other_attack)
        other_creature.apply_damage(own_attack)
        if self.verbose > 0:
            print(
                f"Creature {self}, cnt={self.species_cnt} attacks \n {other_creature}, cnt={other_creature.species_cnt} "
                f"corresponding damage {own_attack} and {other_attack}")

        self.consume_food(0.5)
        other_creature.consume_food(0.25)
        self.movement_points -= 1.0

    def get_current_obs(self):
        """
        Function to get current unprocessed observations of the creature. Must be implemented for all creatures.
        :return: numpy 1D array containing current observations
        """

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
                int(has_vegetation),  # TODO another input must be given such as "has food".
                                        # Vegetation does not necessary mean food for carnivourns
                # self._get_movement_difficulty(tile),  # reserved for movement cost. Probably needs normalization
                self.get_movement_difficulty(self.tile, tile),
                creature_id,  # Other creature id if it is presented at the tile
                creature_cnt,  # Other creature count if it is presented at the tile
            ]

            obs.extend(sub_obs)

        obs = np.array(obs)
        return obs

    def _update_obs(self) -> None:
        """
        Updates self.observation variable, implements frame stack and other observation processings,
        but normalization
        :return:
        """
        if self._memory_frame_stack == 1:
            self.observation = self.get_current_obs()
        else:
            self._observation_buffer.append(np.array(self.get_current_obs()))
            self.observation = np.concatenate(self._observation_buffer)

    def _eat(self, relative_tile_pos: tuple[int, int]):

        if relative_tile_pos == (0, 0):
            self.movement_points -= 1.0
        else:
            self.movement_points -= 2.0
        tile_to_eat = self.tile.world.get_tile_by_index(self.tile.in_map_position, relative_tile_pos)
        has_eaten_food = tile_to_eat.eat_from_tile(self.CONSUMABLE_FOOD_TYPES)

        if self.verbose > 0:
            print(f"eating at {relative_tile_pos}, success = {has_eaten_food}")
        if has_eaten_food:
            self.current_food = min(self.current_food + 5.0, self.MAX_FOOD_SUPPLY)
        else:
            self.consume_food(0.05)

    def _sleep(self, has_done_action):
        if has_done_action:
            if self.verbose > 0:
                print("sleeping... Cannot sleep as an action has already been made")
            self.consume_food(0.005, enable_heal=False)  # small penalty for making restricted action
            return
        if self.verbose > 0:
            print("sleeping...")
        self.consume_food(0.05, enable_heal=True)
        self.movement_points = self.MAX_MOVEMENT_POINTS

    def _split_or_attack(self, relative_tile_pos: tuple[int, int]):
        """ If tile is empty splits half of the creature on the new tile. Otherwise attacks ANY creature on the tile"""
        if relative_tile_pos == (0, 0):  # no action is reserved for splitting into own tile
            self.current_food -= 0.05
            self.movement_points -= 0.1
            return

        new_tile = self.world.get_tile_by_index(self.tile.in_map_position, relative_tile_pos)
        other_creature = self.world.get_creature_on_tile(new_tile)
        # if self is other_creature:  # checking if the creature attacks itself
        #     if self.verbose > 0:
        #         print("suicide attempt (creature is trying to attack itself")
        #     other_creature = None

        if other_creature is None:
            if self.species_cnt > 1:
                self.current_food -= 0.05  # subtracted from both creatures
                self.movement_points -= 1.0

                new_creature = self.__class__(self._agent,
                                              self.tile,
                                              texture=self.texture,
                                              verbose=self.verbose,
                                              creature_to_copy=self)

                remaining_species_cnt = self.species_cnt // 2
                moved_species_cnt = self.species_cnt - remaining_species_cnt
                new_creature.set_tile(new_tile)
                new_creature.species_cnt = moved_species_cnt
                self.species_cnt = remaining_species_cnt
                new_creature_position = (self.tile.in_map_position[0] + relative_tile_pos[0],
                                         self.tile.in_map_position[1] + relative_tile_pos[1])
                self.world.add_creature(new_creature, new_creature_position)
            else:
                self.current_food -= 0.05
                self.movement_points -= 0.1
        else:
            self._attack(other_creature)  # attack already consumes food and movement

    def _move(self, relative_tile_pos: tuple[int, int]):

        if self.verbose > 0:
            print(f"moving to {relative_tile_pos}")
        # new_row = self.tile.in_map_position[0] + relative_tile_pos[0]
        # new_col = self.tile.in_map_position[1] + relative_tile_pos[1]
        new_tile = self.world.get_tile_by_index(self.tile.in_map_position, relative_tile_pos)
        # required_movement = self._get_movement_difficulty(new_tile)
        required_movement = self.get_movement_difficulty(self.tile, new_tile)
        if self.movement_points < required_movement * 0.5:  # attempt to make illegal move
            self.consume_food(0.01)  # small penalty

        other_creature = self.world.get_creature_on_tile(new_tile)
        if other_creature is None:
            self.tile = new_tile  # don't use set_tile function here because it will update obs twice and do unnecessary rescale
            self.movement_points = max(0.0, self.movement_points - required_movement)
            self.consume_food(required_movement / 20.0)
        else:

            if other_creature.CREATURE_ID == self.CREATURE_ID:
                if self.species_cnt == self.MAX_SPECIES_CNT or other_creature == self.MAX_SPECIES_CNT:
                    self.consume_food(0.005)
                    self.movement_points -= 0.1
                if self.species_cnt + other_creature.species_cnt <= self.MAX_SPECIES_CNT:  # creatures fully merge
                    other_creature.species_cnt += self.species_cnt
                    self.movement_points = 0.0  # to block any further action
                    self.world.remove_creature(self)
                else:
                    remaining_species = self.species_cnt + other_creature.species_cnt - self.MAX_SPECIES_CNT
                    other_creature.species_cnt = self.MAX_SPECIES_CNT
                    self.species_cnt = remaining_species
                    self.consume_food(0.05)
                    self.movement_points -= 0.5
            else:
                self._attack(other_creature)  # attack already consumes food and movement

    def _action_to_idx(self, action):
        """ Converts a complex action structure into its index """
        (xy, z) = action  # converting action back to index
        x, y = xy
        x += 1
        y += 1
        action_idx = x * 9 + y * 3 + z  # TODO convert properly to action number
        return np.array([action_idx])

    def _idx_to_action(self, action_idx):
        """ Converts action's index into a complex action structure """
        x, y, action = self.actions[action_idx]  # self.actions[action_idx]
        return ((x, y), action)
