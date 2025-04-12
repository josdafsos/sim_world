import random

import pygame
import numpy as np

class Creature:
    MAX_HP: float = 10.0
    MAX_MOVEMENT_POINTS: float = 10.0
    MAX_FOOD_SUPPLY: float = 10.0
    CONSUMABLE_FOOD_TYPES: tuple[str, ...] = ("grass",)  # "fruit", "meat", "rotten meat"
    MAX_SPECIES_CNT: int = 10  # species amount the tile cannot overreach this value
    CREATURE_ID: int = 1  # Zero 0 reserved for no creature. Used by agents to distinguish between different species.
    VISION_DISTANCE: int = 1  # defines how far the creature can see
    DAYS_TO_BODY_BE_ROTTEN = 15  # body disappears after this number of days
    CHANCE_OF_BIRTH = 0.005  # on new day if the creature is in good condition there is chance to increase
    # population per each existing creature in stack

    # activity that can be done towards selected tile,
    # options: move (also unify, reproduce, attack (if hostile), move onto the current tile = sleep); eat;
    # attack/split (if tile is empty, half of species go to the tile. If own tile is selected,
    # splits part of the species into random nearby location if free.
    # If the selected tile is occupied, the occupying creature is attacked even against friendly)
    AVAILABLE_ACTIONS: tuple[str, ...] = ("move", "eat", "split/attack")   #, "split")  # split/attack is not implemented yet

    def __init__(self, agent,
                 self_tile=None,
                 texture: str | tuple[int, int, int] = (255, 50, 50),
                 verbose: int = 0,
                 creature_to_copy=None):
        """
        Base class for any creautre

        :param agent: any agent of class Agent to control the creature
        :param self_tile: tile on which creature is spawned. Can be None
        :param texture: picture or a color tuple
        :param verbose: 0 - no console prints, 1 - to print states of the creature
        :param creature_to_copy: a creature can be copied, does not override new agent, texture or a tile
        """
        if creature_to_copy is None:
            self.current_hp: float = self.MAX_HP
            self.movement_points: float = self.MAX_MOVEMENT_POINTS
            self.current_food: float = 0.85 * self.MAX_FOOD_SUPPLY  # opposite of hunger value
            self.species_cnt: int = 5  # current number of species at the tile
            self.observation = None
            self.days_since_death: int = 0  # if positive sets the creature to the dead state. First day is the day of death
            self.species_cnt_change: int = 0  # indicates change in size of the stack for birth/death. Reset on new action.
            # Does not indicate split
        else:
            self.current_hp: float = creature_to_copy.current_hp
            self.movement_points: float = creature_to_copy.movement_points
            self.current_food: float = creature_to_copy.current_food
            self.species_cnt: int = creature_to_copy.species_cnt
            self.observation = creature_to_copy.observation
            self.days_since_death: int = creature_to_copy.days_since_death
            self.species_cnt_change: int = 0


        self.agent = agent
        self.tile = self_tile
        self.verbose = verbose
        self.world = None



        if isinstance(texture, str):
            self.texture = pygame.image.load("textures//" + texture).convert_alpha()
        else:
            self.texture = texture  # can be texture or a color of a square
        self.scaled_texture = None

        if self_tile is not None:
            self.set_tile(self_tile)

    def set_tile(self, tile):
        self.tile = tile
        self.world = self.tile.world
        self.observation = self._get_obs()
        self.on_rescale()  # if texture exists scaling it

    def clone_creature(self, creature):
        new_creature = self.__class__(self.agent,
                                      self.tile,
                                      texture=self.texture,
                                      verbose=self.verbose,
                                      creature_to_copy=creature)
        return new_creature

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

        for i in range(self.species_cnt):
            if (self.MAX_SPECIES_CNT > self.species_cnt > 1
                    and self.current_hp > 0.9 * self.MAX_HP
                    and self.current_food > 0.75 * self.MAX_FOOD_SUPPLY
                    and random.random() < self.CHANCE_OF_BIRTH):
                self.species_cnt += 1


        return False

    def make_action(self):

        if self.days_since_death > 0:
            # TODO implement activities happening with the body
            return
        has_done_action = False

        while True:
            # action tuple (tile relative number {(-1, -1), (1,0), (-1,1), etc}, action number)
            action = self.agent.predict(self.observation)
            if self.AVAILABLE_ACTIONS[action[1]] == "move":
                if action[0] == (0, 0):
                    self._sleep(has_done_action)
                    break
                else:
                    self._move(action[0])
            elif self.AVAILABLE_ACTIONS[action[1]] == "eat":
                self._eat(action[0])
            elif self.AVAILABLE_ACTIONS[action[1]] == "split/attack":
                self._split_or_attack(action[0])
            else:
                print(f"Warning, unknown action is called, action index: {action[1]}")

            new_obs = self._get_obs()
            self.agent.learn(self.observation, new_obs, action)
            self.observation = new_obs
            self.species_cnt_change = 0  # restoring the indication of death/birth
            has_done_action = True
            if self.movement_points < 1e-4:
                break

    def on_rescale(self):
        """ This function must be called if the width of Tile has changed"""
        if not isinstance(self.texture, tuple) and self.world is not None:
            width = self.world.tile_width
            self.scaled_texture = pygame.transform.scale(self.texture,
                                                         (0.5 * width, 0.5 * width))

    def draw(self, screen, pos, width, height_scale=None, height_pos=None):
        height_pos = [pos[0] + self.tile.height_level * self.world.height_direction[0] * self.world.height_scale,
                      pos[1] + self.tile.height_level * self.world.height_direction[1] * self.world.height_scale]

        if isinstance(self.texture, tuple):
            if self.days_since_death > 0:
                # drawing a dead body
                pygame.draw.rect(screen, (30, 30, 30), (height_pos[0], height_pos[1], 0.5 * width, 0.5 * width))
            else:
                pygame.draw.rect(screen, self.texture, (height_pos[0] + 0.25 * width,
                                                        height_pos[1] + 0.45 * width,
                                                        0.5*width,
                                                        0.5*width))
        else:
            if self.days_since_death > 0:
                pygame.draw.rect(screen, (30, 30, 30), (height_pos[0], height_pos[1], 0.5 * width, 0.5 * width))
            else:
                self.world.screen.blit(self.scaled_texture, (height_pos[0] + 0.25 * width, height_pos[1] + 0.45 * width))
        if self.days_since_death < 1:
            text_surface = self.world.font.render(str(self.species_cnt), True, (255, 255, 255))
            self.world.screen.blit(text_surface, (height_pos[0] + 0.5 * width, height_pos[1] + 0.45 * width))

    def _get_movement_difficulty(self, tile_to_move):
        movement_difficulty = 2.0
        if self.tile.water.relative_height > 1e-5:
            movement_difficulty += 2.0
        elif self.tile.water.moisture_level > self.tile.water.MOISTURE_LEVEL_TO_RISE_HEIGHT * 0.3:
            movement_difficulty += 1.0
        if tile_to_move.water.relative_height > 1e-5:
            movement_difficulty += 2.0
        elif tile_to_move.water.moisture_level > self.tile.water.MOISTURE_LEVEL_TO_RISE_HEIGHT * 0.3:
            movement_difficulty += 1.0
        if self.tile.height_level - tile_to_move.height_level > 0:
            movement_difficulty -= 0.5
        else:
            movement_difficulty += 1.0

        return movement_difficulty

    def apply_damage(self, damage: float):
        self.current_hp -= damage
        if self.current_hp < 0 and self.days_since_death == 0:
            if damage < 1:
                if random.random() < damage:
                    dead_cnt = 1
                else:
                    dead_cnt = 0
            else:
                dead_cnt = random.randint(1, round(damage))
            self.species_cnt_change = dead_cnt
            self.species_cnt -= dead_cnt
            if self.species_cnt < 1:
                self.days_since_death = 1
                print("the creature died")

    def apply_heal(self, hp_restored: float):
        self.current_hp = min(self.MAX_HP, self.current_hp + hp_restored)

    def _consume_food(self, food_consumed, enable_heal=False):
        self.current_food -= food_consumed
        if self.current_food < 0:
            self.apply_damage(abs(self.current_food))
            self.current_food = 0.0
        elif enable_heal:
            self.apply_heal(food_consumed)

    def _get_obs(self):
        # first set of own states, next state of the tiles in vision range
        obs = [
            self.CREATURE_ID,  # will be useful when meeting other creatures
            self.current_hp,
            self.current_food,
            self.movement_points,
            self.species_cnt,
            self.species_cnt_change,
        ]
        nearby_tiles = self.tile.get_surrounding_tile(self.VISION_DISTANCE)
        for tile in nearby_tiles:
            has_vegetation, _ = tile.has_vegetation(self.CONSUMABLE_FOOD_TYPES)
            creature = self.world.get_creature_on_tile(tile)
            if creature is None:
                creature_id = 0
                creature_cnt = 0
            else:
                creature_id = creature.CREATURE_ID
                creature_cnt = creature.species_cnt
            sub_obs = [
                int(has_vegetation),
                self._get_movement_difficulty(tile),  # reserved for movement cost
                creature_id,  # Other creature id if it is presented at the tile
                creature_cnt,  # Other creature count if it is presented at the tile
            ]

            obs.extend(sub_obs)

        return obs
        
    def _eat(self, relative_tile_pos: tuple[int, int]):

        if relative_tile_pos == (0, 0):
            self.movement_points -= 1.0
        else:
            self.movement_points -= 2.0
        tile_to_eat = self.tile.world.get_tile_by_index(self.tile.in_map_position, relative_tile_pos)
        has_eaten_food = tile_to_eat.consume_vegetation(self.CONSUMABLE_FOOD_TYPES)
        if self.verbose > 0:
            print(f"eating at {relative_tile_pos}, success = {has_eaten_food}")
        if has_eaten_food:
            self.current_food = min(self.current_food + 5.0, self.MAX_FOOD_SUPPLY)
        else:
            self._consume_food(0.05)

    def _sleep(self, has_done_action):
        if has_done_action:
            if self.verbose > 0:
                print("sleeping... Cannot sleep as an action has already been made")
            self._consume_food(0.005, enable_heal=False)  # small penalty for making restricted action
            return
        if self.verbose > 0:
            print("sleeping...")
        self._consume_food(0.05, enable_heal=True)
        self.movement_points = self.MAX_MOVEMENT_POINTS

    def _split_or_attack(self, relative_tile_pos: tuple[int, int]):
        """ If tile is empty splits half of the creature on the new tile. Otherwise attacks ANY creature on the tile"""
        new_tile = self.world.get_tile_by_index(self.tile.in_map_position, relative_tile_pos)
        other_creature = self.world.get_creature_on_tile(new_tile)
        if other_creature is None:
            if self.species_cnt > 1:
                self.current_food -= 0.05  # subtracted from both creatures
                self.movement_points -= 1.0

                new_creature = self.clone_creature(self)
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
            pass  # TODO implement attacking

    def _move(self, relative_tile_pos: tuple[int, int]):

        if self.verbose > 0:
            print(f"moving to {relative_tile_pos}")
        # new_row = self.tile.in_map_position[0] + relative_tile_pos[0]
        # new_col = self.tile.in_map_position[1] + relative_tile_pos[1]
        new_tile = self.world.get_tile_by_index(self.tile.in_map_position, relative_tile_pos)
        required_movement = self._get_movement_difficulty(new_tile)
        if self.movement_points < required_movement * 0.5:  # attempt to make illegal move
            self._consume_food(0.01)  # small penalty

        other_creature = self.world.get_creature_on_tile(new_tile)
        if other_creature is None:
            self.tile = new_tile
            self.movement_points = max(0.0, self.movement_points - required_movement)
            self._consume_food(required_movement / 20.0)
        else:

            if other_creature.CREATURE_ID == self.CREATURE_ID:
                if self.species_cnt == self.MAX_SPECIES_CNT or other_creature == self.MAX_SPECIES_CNT:
                    self._consume_food(0.005)
                if self.species_cnt + other_creature.species_cnt <= self.MAX_SPECIES_CNT:  # creatures fully merge
                    other_creature.species_cnt += self.species_cnt
                    self.world.delete_creature(self)
                    self.movement_points = 0.0  # to block any further action
                else:
                    remaining_species = self.species_cnt + other_creature.species_cnt - self.MAX_SPECIES_CNT
                    other_creature.species_cnt = self.MAX_SPECIES_CNT
                    self.species_cnt = remaining_species
                    self._consume_food(0.05)
                    self.movement_points -= 0.5

            else:
                pass  # TODO only peacefull option is implemented yet


    