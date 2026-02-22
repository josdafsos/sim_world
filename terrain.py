import random
import time
import warnings

import pygame
import numpy as np

import world_properties
from world_properties import WorldProperties
import vegetation
from creatures import Creature
from agents.agents import Agent
import tile_items
from tile_items import TileItem
import utils

# Define neighborhood shifts, used for computations of neighboring tiles effects
# shifts = ((-1, 1), (0, 1), (1, 1),
#           (-1, 0), (1, 0),
#           (-1, -1), (0, -1), (1, -1))
# simplified shifts for only horizontal and vertical interactions
shifts = ((0, -1), (0, 1),
          (-1, 0), (1, 0),)


class Terrain:
    height_direction = (np.cos(np.deg2rad(210)),
                        np.sin(np.deg2rad(210)))  # with the height increase tile is move towards this vector
    HEIGHT_SCALE_COEFF = 0.75  # defines offset of tall object from the small ones
    HOURS_PER_STEP = 6  # how many hours are passed after each mini-step is made

    def __init__(self,
                 size: tuple[int, int],
                 screen=None,
                 enable_visualization: bool = True,
                 enable_map_loop_visualization: bool = False,
                 verbose: int = 0,
                 is_round_map: bool = True,
                 generation_method: str = 'random',
                 reset_on_dead_world: bool = True,
                 steps_to_reset_world: int = -1,
                 creatures_to_respawn: tuple[tuple[Creature, Agent, int], ...] | None = None):
        """
        :param size - (width, heigh) size of the generated map
        :param screen - pygame screen on which the world will be drawn; if None nothing will be drawn
        :param verbose - {0, 1, 2} sets the amount of information printed in the console
        :param is_round_map If true the map connects opposite ends of the map
        :param generation_method: a method to create a new world.
        String, following options are available:
        random - (default) fully random world,
        consistent_random - random world with gradual changes in height and tile content
        :param reset_on_dead_world - if True the world is monitored for being "dead", i.e without any vegetation.
        In which case it will be recreated. New random generation is made for corresponding generation options
        :param steps_to_reset_world: if positive, the world will be reset after the given amount of steps.
        Note - there is a small margin of error steps after which the world will be reset
        :param creatures_to_respawn - the world is monitored for the amount of specified creatures,
        if the current creatures number is lower than the given integer, then additional creatures will be spawned
        with a given agent.
        Can be None, then nothing will be spawned.
        Can be tuple of tuples of (Creature Class, agent instance, minimum count of creatures in the world,
        {kwargs for creature initialization})
        """

        # --- visual settings ---
        self.screen = screen
        self.camera_pos = [0, 0]
        self.camera_visible_x_range = []  # coordinates of start and end of visible tiles along x directions
        self.camera_visible_y_range = []  # coordinates of start and end of visible tiles along y directions
        self.camera_visible_tiles_cnt = 0  # number of tiles visible on the screen at the moment
        self.tile_width = 30
        self.height_scale = self.tile_width * self.HEIGHT_SCALE_COEFF  # original self.tile_width * 0.5
        self.enable_visualization = enable_visualization
        if self.screen is not None:  # TODO a graphics init func must be implemented, which will update setting on screen change
            self.font = pygame.font.SysFont(None, 48)

        # --- map settings ---
        self.is_round_map = is_round_map
        self.enable_map_loop_visualization = enable_map_loop_visualization and is_round_map
        self.map_size = size
        self.terrain_map: list[list[Tile | ...]] = [[None for _ in range(self.map_size[1])] for _ in range(self.map_size[0])]
        self.map_generation_method: str = generation_method

        self.height_mat = np.zeros(self.map_size)
        self.water_source_mat = np.zeros(self.map_size)
        self.moisture_level_mat = np.zeros(self.map_size)
        # wrapped padding on sides to reflect wrapped world
        self.pad_moisture_level_mat = np.pad(self.moisture_level_mat, pad_width=1, mode='wrap')
        self.water_relative_height = np.zeros(self.map_size)
        self.absorbtion_coeff_mat = np.zeros(self.map_size)
        self.vegetation_presence_map = np.zeros(self.map_size)  # 1 if any vegetation is presented, otherwise 0. For this case is more convenient than boolean

        # --- other settings ---
        self.verbose: int = verbose  # see param description above
        self.total_steps: int = 0  # counter for world updates (i.e. every 24 hours, not the creature steps), reset() sets it to 0
        self.autoplay: bool = False  # If True, the world steps are called unstoppably. Used externally
        self.current_time_hours: int = 0  # current time of the day
        self.creatures: list[Creature, ...] = []  # list of all creatures currently living in the world
        self.is_new_step_visual: bool = True  # Set to true every time when the world has just made a step. Used for visual updates
        self.items_prepare_step_list = []  # list of items that have HAS_PREPARE_STEP == True
        self.items_step_list = []  # list of items that have HAS_STEP == True

        # --- monitoring options ----
        self.reset_on_dead_world: bool = reset_on_dead_world  # flag to check if a "dead" world must be restarted
        self.steps_to_reset_world: int = steps_to_reset_world
        self.creatures_to_respawn: tuple[tuple[Creature, Agent, int, dict], ...] | None = creatures_to_respawn
        self.MONITORING_PERIOD: int = 10  # the interval of simulation steps at which monitoring occurs (checking for dead world and respawning creatures)

        self.reset()

    def reset(self) -> None:
        """
        Resets simulation, recreates world. Any of random generation options causes a new random world generation.
        Autoplay option is untouched,
        """
        self.total_steps = 0
        self.creatures = []
        self.items_prepare_step_list = []
        self.items_step_list = []

        self.terrain_map = [[None for _ in range(self.map_size[1])] for _ in range(self.map_size[0])]
        self.height_mat = np.zeros(self.map_size)
        self.water_source_mat = np.zeros(self.map_size)
        self.moisture_level_mat = np.zeros(self.map_size)
        # wrapped padding on sides to reflect wrapped world
        self.pad_moisture_level_mat = np.pad(self.moisture_level_mat, pad_width=1, mode='wrap')
        self.water_relative_height = np.zeros(self.map_size)
        self.absorbtion_coeff_mat = np.zeros(self.map_size)
        self.vegetation_presence_map = np.zeros(self.map_size)

        for row in range(self.map_size[0]):
            for col in range(self.map_size[1]):
                surrounding_tiles, _ = self._get_surrounding_tiles(row, col, self.terrain_map, search_diameter=3)
                self.terrain_map[row][col] = Tile(self, surrounding_tiles,
                                                  (row, col),
                                                  self.map_generation_method)
        for row in range(self.map_size[0]):
            for col in range(self.map_size[1]):
                _, distance_sorted_tiles = self._get_surrounding_tiles(row, col, self.terrain_map, search_diameter=5)
                self.terrain_map[row][col]._set_surrounding_tiles(distance_sorted_tiles)
        self.terrain_map = tuple(tuple(inner) for inner in self.terrain_map)
        self.camera_move("update")

    def monitor(self) -> None:
        """
        Checks if the world's state meets following conditions:
        - "dead" world
        - max world steps reached
        - spawns missing creatures
        """

        # checking dead world
        if self.reset_on_dead_world:
            # NOTE: The following condition checks for all plant types. However, some newly added plant types
            # might not be useful to make world "living", i.e. to support animals as food. Thus, this condition must be
            # re-iterated in the future
            if np.sum(self.vegetation_presence_map) < 1:  # no living plant
                print("The world has no vegetation, resetting...")
                self.reset()
                return
        if 0 < self.steps_to_reset_world < self.total_steps:
            print(f"Maximum world steps of {self.steps_to_reset_world} reached, resetting...")
            self.reset()
            return

        # checking creatures count
        if self.creatures_to_respawn is not None:
            creatures_cnt_dict = {}
            for creature in self.creatures:
                creatures_cnt_dict[creature.TYPE] = creatures_cnt_dict.get(creature.TYPE, 0) + creature.species_cnt
            for spawn_creature in self.creatures_to_respawn:
                if (spawn_creature[0].TYPE not in creatures_cnt_dict or
                        creatures_cnt_dict[spawn_creature[0].TYPE] < spawn_creature[2]):
                    self.add_creature(spawn_creature[0](spawn_creature[1], **spawn_creature[3]))

    def add_creature(self, creature: Creature, position: None | tuple[int, int] = None) -> None:
        """
        :param creature:
        :param position: coordinates on the map or None for random position
        :return:
        """
        if position is None:
            row = random.randint(0, self.map_size[0])
            col = random.randint(0, self.map_size[1])
            position = (row, col)

        tile = self.get_tile_by_index(position)
        creature.set_tile(tile)
        self.creatures.append(creature)

    def add_active_item(self, item: TileItem):
        """
        Adds tile item to the world, only adds active items, i.e. those that are executed on step or prepare step.
        Passive items will not be added
        :param item:
        """
        if item.HAS_PREPARE_STEP:
            self.items_prepare_step_list.append(item)
        if item.HAS_STEP:
            self.items_step_list.append(item)

    def delete_active_item(self, item: TileItem) -> None:
        """
        Deletes a give item from the world, shows Warning if such item does not exist
        :param item:
        """
        is_deleted = False  # flag to check if the item was deleted from any of the lists
        if item in self.items_prepare_step_list:
            self.items_prepare_step_list.remove(item)
            is_deleted = True
        if item in self.items_step_list:
            self.items_step_list.remove(item)
            is_deleted = True
        if not is_deleted:
            warnings.warn("Request to remove an un-existing TileItem")

    def prepare_step_vegetation_mat(self):
        vegetation_idx = np.argwhere(self.vegetation_presence_map > 0.5)  # indexes of all tiles with vegetation
        # TODO the list can be saved and elements can be erased and included instead of list re-creation every step. Should work faster
        # TODO probably a list of tiles containing vegetation would be a simpler solution than a list of all vegetation
        for x, y in vegetation_idx:
            tile = self.terrain_map[x][y]
            for vegetable in tile.vegetation_list:
                if vegetable is not None:
                    vegetable.prepare_step_vegetation()
            # for key in list(tile.vegetation_dict.keys()):  # TODO seems to consume too much time. Find another way to iterate + ocasional delete
            #     # one plant can delete another in the middle of this cycle
            #     # that's why the key validation condition is needed
            #     if key in tile.vegetation_dict:
            #         tile.vegetation_dict[key].prepare_step_vegetation()

    def step_vegetation_mat(self):
        vegetation_idx = np.argwhere(self.vegetation_presence_map > 0.5)  # indexes of all tiles with vegetation
        # TODO same comment as for prepare step - create and modify single vegetation list instead of re-creating it on each loop
        for x, y in vegetation_idx:
            tile = self.terrain_map[x][y]
            for vegetable in tile.vegetation_list:
                if vegetable is not None:
                    vegetable.step_vegetation()
            # for key in list(tile.vegetation_dict.keys()):  # TODO the list generation here takes a lot of time
            #     tile.vegetation_dict[key].step_vegetation()

    def step_water_mat(self):
        """ Computes water physics """
        # TODO limit max water level
        # TODO there seems to be a bug after many simulation steps where water visually disappears, but seem to be
        # still present

        water_abs_height = self.height_mat + self.water_relative_height
        pad_abs_water_level = np.pad(water_abs_height, pad_width=1, mode='wrap')  # wrapping for round world. TODO no wraping option for a boundary world

        absorption = self.absorbtion_coeff_mat * (1 - 0.95 * self.height_mat)
        absorption *= (1 - 0.5 * self.vegetation_presence_map)

        flow_out_cnt_mat = np.ones_like(self.height_mat)
        flow_out = np.zeros_like(self.height_mat)
        flow_in = np.zeros_like(self.height_mat)

        for dy, dx in shifts:
            neighbor_abs_water_height = pad_abs_water_level[1 + dy: 1 + dy + self.map_size[0], 1 + dx: 1 + dx + self.map_size[1]]
            water_diff = neighbor_abs_water_height - water_abs_height
            # Where the neighbor is lower than current tile → water flows out
            out_mask = water_diff < 0
            flow_out += out_mask * self.moisture_level_mat * Water.FLOW_RATE_TIME_CONSTANT
            flow_out_cnt_mat += out_mask  # counting to how many neighbouring tiles the water is floating

        flow_out = flow_out / flow_out_cnt_mat

        # making pre-computations for the next for_loop
        flow_out_rate = self.moisture_level_mat * Water.FLOW_RATE_TIME_CONSTANT / flow_out_cnt_mat
        pad_flow_out_rate = np.pad(flow_out_rate, pad_width=1, mode='wrap')

        for dy, dx in shifts:
            neighbor_abs_water_height = pad_abs_water_level[1 + dy: 1 + dy + self.map_size[0], 1 + dx: 1 + dx + self.map_size[1]]
            neighbor_flow_out_rate = pad_flow_out_rate[1 + dy: 1 + dy + self.map_size[0], 1 + dx: 1 + dx + self.map_size[1]]
            water_diff = neighbor_abs_water_height - water_abs_height
            in_mask = water_diff > 0
            flow_in += in_mask * neighbor_flow_out_rate

        flow_difference = flow_in - flow_out
        self.moisture_level_mat += self.water_source_mat + flow_difference - absorption
        self.moisture_level_mat[self.moisture_level_mat < 1e-4] = 0  # cutting low water level to zero
        self.moisture_level_mat[self.moisture_level_mat > 1.0] = 1.0  # limiting maximum water level
        self.pad_moisture_level_mat = np.pad(self.moisture_level_mat, pad_width=1, mode='wrap')

        self.water_relative_height = np.maximum((self.moisture_level_mat - Water.MOISTURE_LEVEL_TO_RISE_HEIGHT) *
                                                Water.MOISTURE_TO_WATER_LEVEL_COEFF, 0)

        # Computing erosion due to flow
        height_diff = np.tanh(flow_difference) * 3e-4  # limiting the maximum height difference per step
        eroding_tiles = np.abs(height_diff) > 1e-7
        intaking_water_tiles = flow_out < 1e-8  # tiles that intake water, but it does not flow further
        flow_through_tiles = np.bitwise_not(intaking_water_tiles)  # tiles that have a through flow

        intaking_water_tiles_idx = np.argwhere(np.bitwise_and(intaking_water_tiles, eroding_tiles))
        flow_through_tiles_idx = np.argwhere(np.bitwise_and(flow_through_tiles, eroding_tiles))
        for x, y in intaking_water_tiles_idx:
            self.terrain_map[x][y].change_height_and_content(-height_diff[x, y] * 0.01, 'sand')
        for x, y in flow_through_tiles_idx:
            self.terrain_map[x][y].change_height_and_content(-np.abs(height_diff[x, y]), 'rock')

    def step(self) -> None:
        time_before_step = time.time()

        if self.creatures and self.current_time_hours <= 24 - self.HOURS_PER_STEP:  # in case we have any creatures, iterate through them
            tmp_creatures_list = self.creatures.copy()  # because some of the creatures may die during actions
            # and will be excluded from self.creatures list
            for creature in tmp_creatures_list:
                creature.make_actions()
            self.current_time_hours += self.HOURS_PER_STEP
        else:
            self.current_time_hours = 0
            self.is_new_step_visual = True  # set to False in draw call TODO set it false

            # --- prepare step section ---
            self.prepare_step_vegetation_mat()

            for item in self.items_prepare_step_list:
                item.prepare_step()

            # --- step section ---
            for item in self.items_step_list:
                item.step()

            self.step_water_mat()  # new water physics computation, done in single step without preparation
            self.step_vegetation_mat()

            for creature in self.creatures:
                creature.new_day()

            if self.total_steps % self.MONITORING_PERIOD == 0:
                self.monitor()

            self.total_steps += 1

        if self.verbose > 0:
            print(f"--- Step {self.total_steps} was made in {time.time() - time_before_step:.4f} seconds ----")

    def multiple_steps(self, steps_cnt):
        enable_visualization = self.enable_visualization
        self.enable_visualization = False
        for _ in range(steps_cnt):
            self.step()
        self.enable_visualization = enable_visualization

    def get_tile_by_index(self, tile_index: tuple[int, int], offset=(0, 0)):
        if self.is_round_map:
            row = (tile_index[0] + offset[0] + self.map_size[0]) % self.map_size[0]
            col = (tile_index[1] + offset[1] + self.map_size[1]) % self.map_size[1]
            return self.terrain_map[row][col]
        else:
            raise "Bordered maps are not implemented yet"

    def update_tile_height(self, tile_coords: tuple, new_height: float):
        self.height_mat[tile_coords[0], tile_coords[1]] = new_height

    def update_water_source(self, tile_coords: tuple, new_water_source: float):
        self.water_source_mat[tile_coords[0], tile_coords[1]] = new_water_source

    def update_absorbtion_coeff(self, tile_coords: tuple, new_absorbtion_coeff: float):
        self.absorbtion_coeff_mat[tile_coords[0], tile_coords[1]] = new_absorbtion_coeff

    def update_vegetation_presence(self, tile_coords: tuple, presented: float):
        """
        :param tile_coords:
        :param presented: float, 1.0 if any vegetation is presented on the tile and 0.0 otherwise
        :return:
        """
        self.vegetation_presence_map[tile_coords[0], tile_coords[1]] = presented

    def get_creature_on_tile(self, tile):
        """ Returns a creature on the tile. If tile is empty returns None"""
        for creature in self.creatures:
            if creature.tile == tile:
                return creature
        return None

    def remove_creature(self, creature: Creature) -> bool:
        """
                Removes creature from the world
                :param creature:
                :return: True if creature was removed successfully, otherwise False (in case creature does not exist)
        """
        if creature in self.creatures:
            self.creatures.remove(creature)
            if self.verbose > 0:
                print(f"creature deleted, total creature count: {len(self.creatures)}")
            return True
        else:
            if self.verbose > 0:
                warnings.warn(f"Attempted to remove a creature that does not exist in the world, creature: {creature}")
            return False

    def _get_surrounding_tiles(self, row: int, col: int, tile_map: list[list[...]], search_diameter=3) -> list[list[...]]:
        """
        Central tile is set to None as the surrounding around it is computed
        """
        surrounding_size = search_diameter  # must be odd
        center = (surrounding_size - 1) // 2
        map_width, map_height = len(tile_map), len(tile_map[0])
        surrounding_tiles = [[None for _ in range(surrounding_size)] for _ in range(surrounding_size)]
        distance_sorted_tiles = {}
        for i in range(surrounding_size):
            for j in range(surrounding_size):
                x = row + i - center
                y = col + j - center
                if x >= map_height:
                    x %= map_width
                if y >= map_width:
                    y %= map_height
                surrounding_tiles[i][j] = tile_map[x][y]
                radius = max(abs(i-center), abs(j-center))
                if radius != 0:
                    if radius not in distance_sorted_tiles:
                        distance_sorted_tiles[radius] = []
                    distance_sorted_tiles[radius].append(tile_map[x][y])

        surrounding_tiles[center][center] = None

        return surrounding_tiles, distance_sorted_tiles

    def camera_move(self, direction: str, is_camera_rescaled: bool = False):
        if self.screen is None:
            return
        step = 0.5 * self.tile_width

        if direction == "left":
            self.camera_pos[0] += step
        elif direction == "right":
            self.camera_pos[0] -= step
        elif direction == "up":
            self.camera_pos[1] += step
        elif direction == "down":
            self.camera_pos[1] -= step
        elif direction == "zoom in":
            self.tile_width *= 1.05
            is_camera_rescaled = True
        elif direction == "zoom out":
            self.tile_width *= 0.95
            is_camera_rescaled = True
        elif direction == "update":  # request to only update camera settings
            pass
        else:
            raise "unknown camera motion direction"

        self.height_scale = self.tile_width * self.HEIGHT_SCALE_COEFF
        if is_camera_rescaled:
            if self.tile_width < 30:
                self.font = pygame.font.SysFont(None, 30)
            else:
                self.font = pygame.font.SysFont(None, 48)

            for creature in self.creatures:
                creature.on_rescale()  # changing texture sizes
            vegetation_idx = np.argwhere(self.vegetation_presence_map > 0.5)  # indexes of all tiles with vegetation
            for x, y in vegetation_idx:
                for veg_item in self.terrain_map[x][y].vegetation_list:
                    if veg_item is not None:
                         veg_item.on_rescale()

            # TODO also rescale texture for tiles (texturing is not implemented there yet)

        start_x = -int(self.camera_pos[0] / self.tile_width)
        end_x = int(start_x + self.screen.get_width() / self.tile_width + 1)
        start_y = -int(self.camera_pos[1] / self.tile_width)
        end_y = int(start_y + self.screen.get_height() / self.tile_width + 1)
        if not self.enable_map_loop_visualization:
            start_x = max(start_x, 0)
            end_x = min(end_x, self.map_size[0])
            start_y = max(start_y, 0)
            end_y = min(end_y, self.map_size[1])

        self.camera_visible_x_range = (start_x, end_x)
        self.camera_visible_y_range = (start_y, end_y)
        self.camera_visible_tiles_cnt = (end_x - start_x) * (end_y - start_y)

    def camera_fit_view(self):
        if self.screen is None:
            return
        screen_width = self.screen.get_width()
        screen_height = self.screen.get_height()
        horizontal_size = screen_width / self.map_size[0]
        vertical_size = screen_height / self.map_size[1]
        self.tile_width = min(horizontal_size, vertical_size) * 0.98
        self.camera_pos = [0, 0]
        self.camera_move("update", is_camera_rescaled=True)

    def draw(self):
        if not self.enable_visualization or self.screen is None:
            return

        # case of map loop visualization disabled
        tiles_drawn = 0
        if not self.enable_map_loop_visualization:
            for row in range(self.camera_visible_x_range[0], self.camera_visible_x_range[1]):
                for col in range(self.camera_visible_y_range[0], self.camera_visible_y_range[1]):
                    cur_pos = [self.camera_pos[0] + self.tile_width * row, self.camera_pos[1] + self.tile_width * col]
                    self.terrain_map[row][col].draw(self.screen, cur_pos, self.tile_width, self.is_new_step_visual)
                    tiles_drawn += 1
            for creature in self.creatures:
                row, col = creature.tile.in_map_position
                cur_pos = [self.camera_pos[0] + self.tile_width * row, self.camera_pos[1] + self.tile_width * col]
                creature.draw(self.screen, cur_pos)

            self.is_new_step_visual = False
        else:
            print("camera loop visualization is not yet implemented")  # TODO


class Water:
    # common water constants
    MOISTURE_LEVEL_TO_RISE_HEIGHT = 0.8  # if moisture reaches this level, water level starts to rise
    FLOW_RATE_TIME_CONSTANT = 0.5  # defines the speed at which water propagates into nearby tiles
    MAX_SOURCE_OUTPUT = 0.05
    MIN_SOURCE_OUTPUT = 0.001
    MOISTURE_TO_WATER_LEVEL_COEFF = 0.10  # defines the ratio at which water level increases wrt to moisture increase

    def __init__(self, self_tile):
        self.tile = self_tile  # reference to the tile that posses the water
        self.moisture_level: float = 0  # total amount of water on a tile
        self.relative_height = 0  # water starts to gain height with moisture level growing, measured from the tile height
        self.absolute_height = self.tile.height_level + self.relative_height
        self.flow_in = 0
        self.flow_out = 0
        self.moisture_lines = []
        self.source_intensity = 0
        self.absorption_coeff: float = 0.0

    def add_water_source(self, intensity: float):
        self.source_intensity = intensity

    def update_absorbtion_coeff(self):
        self.absorption_coeff = 0.0
        for soil_type in self.tile.content_dict:  # taking absorption proportionally to the soil %
            self.absorption_coeff += (world_properties.soil_types[soil_type]["water absorption"] *
                                      self.tile.content_dict[soil_type])
        # if self.absorption_coeff < 0:  # this check was used to find a bug, no it is fixed
        #     warnings.warn("Negative absorption coefficient detected")
        self.tile.world.update_absorbtion_coeff(self.tile.in_map_position, self.absorption_coeff)

    def prepare_step_water(self):
        flow_cnt_offset = 1
        flow_out_cnt = flow_cnt_offset
        surround_tiles = self.tile.surround_tiles_dict["1"]
        for tile in surround_tiles:
            if tile.water.absolute_height - self.absolute_height < 0:
                flow_out_cnt += 1
        if flow_cnt_offset == flow_out_cnt:  # if we don't have any outgoing streams we don't need to go further
            return

        for tile in surround_tiles:
            water_diff = tile.water.absolute_height - self.absolute_height
            if water_diff < 0:
                single_tile_flow_rate_out = self.moisture_level * Water.FLOW_RATE_TIME_CONSTANT / flow_out_cnt
                tile.water.flow_in += single_tile_flow_rate_out
                self.flow_out += single_tile_flow_rate_out

    def step_water(self, update_visuals):

        absorption_level = self.absorption_coeff * (1 - 0.95 * self.tile.height_level)  # the higher tile, the smaller absorbtion

        if self.tile.world.vegetation_presence_map[self.tile.in_map_position] > 1e-4:  # presence of vegetation_dict reduces water  absorption
            absorption_level *= 0.5

        self.moisture_level += self.source_intensity
        self.moisture_level += self.flow_in - self.flow_out
        self.moisture_level -= absorption_level

        # water starts to fill the tile only when a threshold is reached
        self.relative_height = max((self.moisture_level - Water.MOISTURE_LEVEL_TO_RISE_HEIGHT) *
                                    Water.MOISTURE_TO_WATER_LEVEL_COEFF, 0)
        self.absolute_height = self.tile.height_level + self.relative_height

        self.moisture_lines = []  # for visual effects of moisture
        if self.moisture_level < 0.0001:
            self.moisture_level = 0

        height_diff = np.tanh(self.flow_in - self.flow_out) * 3e-4  # limiting the maximum height difference per step

        if abs(height_diff) > 1e-7:  # condition to avoid no water tiles to change accidentally
            # print("land difference", height_diff)
            if height_diff < 0:
                pass  # initially rock must have been washed out by the stream, but the water was digging well on itself

            else:
                if self.flow_out < 1e-3:  # if it is closer to be a lake making it a sandy bottom
                    self.tile.change_height_and_content(-abs(height_diff) * 0.01, 'sand')
                else:
                    self.tile.change_height_and_content(-abs(height_diff), 'rock')  # if it is a river style, making rocky bottom
                # pass
                # self.tile.change_height_and_content(0.01 * height_diff, 'sand')  # initial approach

        self.flow_in = 0.0
        self.flow_out = 0.0

    def draw(self, screen, pos, width, height_scale, height_pos, new_step: bool) -> None:
        """
        :param screen:
        :param pos:
        :param width:
        :param height_scale:
        :param height_pos:
        :param new_step: bool, Must be True if a world has made a new step on the previous frame. Must be false otherwise
        :return:
        """

        relative_height = self.tile.world.water_relative_height[self.tile.in_map_position]  #  self.relative_height
        moisture_level = self.tile.world.moisture_level_mat[self.tile.in_map_position]  # self.moisture_level

        water_transparency = max(255 - moisture_level * 255, 20)

        water_level_offset = [pos[0] + self.absolute_height * self.tile.world.height_direction[0] * height_scale,
                              pos[1] + self.absolute_height * self.tile.world.height_direction[1] * height_scale]
        if relative_height > 0.0001:
            pygame.draw.rect(screen, (40, 40, 255, water_transparency), (water_level_offset[0], water_level_offset[1], width, width))
        elif self.tile.world.camera_visible_tiles_cnt < 3000:
            if self.source_intensity > 1e-4:
                radius = width/2 * self.source_intensity / self.MAX_SOURCE_OUTPUT
                pygame.draw.circle(screen, (0, 0, 200, 200),
                                   (height_pos[0] + width/2, height_pos[1] + width/2),
                                   radius)

            if moisture_level > 0.0001: #width > 8:  # second condition is for drawing optimization
                if new_step:  # updating water lines on a new step (for motion animation)
                    max_lines = 20
                    lines_cnt = int(moisture_level / Water.MOISTURE_LEVEL_TO_RISE_HEIGHT * max_lines) + 3
                    line_length = 0.2 + lines_cnt / max_lines * 0.4
                    line_width = 1 + int(lines_cnt / max_lines * 4)
                    self.moisture_lines = []
                    for _ in range(lines_cnt):
                        y = random.random()
                        x_start = random.random()
                        x_end = min(x_start + line_length, 1)
                        self.moisture_lines.append([x_start, x_end, y, line_width])

                x_min = height_pos[0]
                y_min = height_pos[1]
                for line in self.moisture_lines:
                    pygame.draw.line(screen,
                                     (0, 0, 255),
                                     (x_min + line[0] * width, y_min + line[2] * width),
                                     (x_min + line[1] * width, y_min + line[2] * width),
                                     line[3])


class Tile:

    # TODO cache tile states to faster obtained by the creatures

    def __init__(self,
                 world: Terrain,
                 surrounding_tiles: list[list[...]],
                 in_map_position: tuple[int, int],
                 generation_method: str):
        """
        Generates random tile
        :param world: instance of Terrain class on which the tile is spawned
        :param surrounding_tiles: list of list of all initialized tiles in the closest neighbourhood
        :param in_map_position: tuple of (row, column) indexes in the world
        :param generation_method: a method to create a new world.
        String, following options are available:
        random - (default) fully random world,
        consistent_random - random world with gradual changes in height and tile content
        """

        self.world = world
        self.content_dict: dict = {}
        self.modifiers: list = []
        self.surround_tiles_dict: dict = {}
        self.in_map_position: tuple[int, int] = in_map_position  # (row, col) indexes of the tile in the world
        # self.vegetation_dict: dict = {}  # str type : instance  TODO should it be replaced with a collection that holds vegetation groups?
        self.vegetation_list: list[vegetation.Vegetation, ...] = utils.VEGETATION_LIST.copy()
        # self.dead_cnt: int = 0  # number of dead bodies on the tile
        # self.dead_creature: Creature | None = None  # temporary variable, will be used for drawing mostly
        self.nutrition = 0

        flattened_tiles = [item for sublist in surrounding_tiles if sublist is not None for item in sublist if
                           item is not None]

        # purely random tile
        self.height_level: float = random.random()  # height of the tile, can lay in range [0, 1]
        tile_content = {}  # TODO is there a sorted data structure instead of dictionary?
        for key in world_properties.soil_types:
            tile_content[key] = random.random()
        highest_content_types = sorted(tile_content, key=tile_content.get, reverse=True)[:2]  # picking two biggest values
        sum_content = 0
        for key in highest_content_types:
            sum_content += tile_content[key]
            self.content_dict[key] = tile_content[key]
        for key in self.content_dict:
            self.content_dict[key] /= sum_content  # normalizing total content to [0, 1]

        if generation_method == 'consistent_random':
            if flattened_tiles:  # checking that currently initialized tile is not the very first
                sum_height = 0
                sum_content = 1.0  # we normalized the tile's content before, that's why it is started with 1.0
                total_content_dict = self.content_dict
                for tile in flattened_tiles:
                    sum_height += tile.height_level
                    for key, value in tile.content_dict.items():
                        total_content_dict[key] = total_content_dict.get(key, 0) + value
                        sum_content += value

                for key in total_content_dict:
                    total_content_dict[key] /= sum_content  # normalizing total content to [0, 1]
                self.content_dict = total_content_dict

                mean_height = sum_height / len(flattened_tiles)
                self.height_level = np.random.normal(mean_height, 0.25)

        self.height_level = max(0.0, self.height_level)  # clipping height level between [0, 1]
        self.height_level = min(1.0, self.height_level)

        self.highest_content_type: str = sorted(tile_content, key=tile_content.get, reverse=True)[0]  # writing the highest content

        self._update_nutrition()  # computing nutrition of the soil based on soils in the tile
        self.water: Water = Water(self)  # wouldn't it be better to inherit water instead of making an instance?!

        for generation_property in world_properties.world_generation_properties["generation probabilities"]:
            if random.random() < generation_property[1]:
                if generation_property[0] == "water source":
                    water_source_intensity = max(random.random()*Water.MAX_SOURCE_OUTPUT,
                                                 Water.MIN_SOURCE_OUTPUT)
                    self.water.add_water_source(water_source_intensity)
                    self.modifiers.append([generation_property[0], water_source_intensity])   # second element is generating speed
                elif generation_property[0] == "grass":
                    vegetation.Grass(self)

                # print("added ", generation_property[0])

        # items
        self.all_items_dict = {}  # dictionary of all items on the tile, TODO it should be another data structure that would consider the drawing priority order
        self.passive_items_dict = {}  # dictionary of passive items (subset of all items list)

        # other
        self.water.update_absorbtion_coeff()
        self.world.update_tile_height(self.in_map_position, self.height_level)
        self.world.update_water_source(self.in_map_position, self.water.source_intensity)

        # visual properties
        self.texture: tuple[float, float, float] | str | None = None  # if None, auto texture will be assigned
        self._set_tile_texture_by_soil()

    def _set_tile_texture_by_soil(self) -> None:
        """
        Updates tile texture based on soil content. Does nothing if custom texture is applied
        Also updates self.highest_content_type variable
        :return:
        """

        self.highest_content_type = sorted(self.content_dict, key=self.content_dict.get, reverse=True)[0]

        if isinstance(self.texture, str):  # if custom texture is applied doing nothing
            # TODO texture must be updated according to the corresponding texture, if it not custom
            return
        self.texture = world_properties.soil_types[self.highest_content_type]["color"]

    def _update_nutrition(self):
        """
        Computes nutrition of the soil based on soils in the tile, updates corresponding variable
        :return: None
        """
        for key in self.content_dict:
            self.nutrition += self.content_dict[key] * world_properties.soil_types[key]["nutritional value"]

    def has_vegetation(self, vegetation_list: tuple[str, ...]):
        """
        Searches for vegetation.TYPE, returns true if any from the input tuple is presented.
        See also has_vegetation_group method
        :param vegetation_list:
        :return:
        """
        has_vegetation = False
        existing_plant_key = None
        if self.world.vegetation_presence_map[self.in_map_position] < 1e-4:
            return has_vegetation, existing_plant_key
        for plant in vegetation_list:
            tile_vegetation = self.vegetation_list[utils.plant_group_index_mapping_dict[
                utils.plant_type_group_mapping_dict[plant]]]
            if tile_vegetation is not None and tile_vegetation.TYPE == plant:
                has_vegetation = True
                existing_plant_key = plant
                break
        return has_vegetation, existing_plant_key

    def has_vegetation_group(self, vegetation_group_list: tuple[str, ...]):
        """
        Searches for vegetation.GROUP, returns true if any from the input tuple is presented
        See also has_vegetation method
        :param vegetation_group_list:
        :return:
        """
        has_vegetation = False
        existing_plant_key = None
        for group in vegetation_group_list:
            if self.vegetation_list[utils.plant_group_index_mapping_dict[group]] is not None:
                has_vegetation = True
                existing_plant_key = group
                break

        return has_vegetation, existing_plant_key

    def consume_vegetation(self, vegetation_options) -> bool:
        has_vegetation, plant_key = self.has_vegetation_group(vegetation_options)
        if not has_vegetation:
            return False

        # del self.vegetation_dict[plant_key]
        self.vegetation_list[utils.plant_group_index_mapping_dict[plant_key]]._delete_plant()
        #self.world.update_vegetation_presence(self.in_map_position, 0.0)
        return True

    def has_meat(self) -> bool:
        """ Returns True if a dead body is present on the tile, returns False otherwise"""
        return tile_items.DeadBody.TYPE in self.all_items_dict

    def eat_from_tile(self, edible_options: tuple) -> bool:
        """
        Eats a food the list, meat consumption is in priority
        :param edible_options: tuple of all things a creature can eat
        :return: True if there was food and it was consumed. Otherwise returns False.
        The return will be replaced with float consumed food nutrition value in future
        """

        # meat consumption case
        if world_properties.MEAT in edible_options and self.has_meat():
            self.all_items_dict[tile_items.DeadBody.TYPE].erase_dead_creature(1)
            return True

        # vegetation consumption
        return self.consume_vegetation(edible_options)

    def change_height_and_content(self,
                                  height_difference: float,
                                  added_soil_type: str | None = None,
                                  same_material_change_coeff: float = 1.0) -> None:
        """
        Modify height of the tile by a given value. Height will be maintained in [0, 1] range
        Soil can be added, proportional to the height difference
        :param height_difference: Value to change the height (positive or negative)
        :param added_soil_type: if not None this content will be added to the tile in proportion with height
        :param same_material_change_coeff [0, 1] scale the change of height proportional to content of the replacing material
        1 - height will not be changed at all if the tile is single material, 0 - material will be change anyway;
        otherwise in-between linear interpolation
        :return:
        """

        same_material_change_coeff = min(1, max(0, same_material_change_coeff))
        desired_soil_relative_content_ratio = 0.0
        if added_soil_type is not None:
            content_sum = 0
            for value in self.content_dict.values():
                content_sum += value

            if added_soil_type not in self.content_dict:
                self.content_dict[added_soil_type] = 0

            added_value = content_sum * abs(height_difference)
            self.content_dict[added_soil_type] += added_value
            total_soil_types = len(self.content_dict)
            for key in list(self.content_dict.keys()):
                if key == added_soil_type:
                    desired_soil_relative_content_ratio = self.content_dict[added_soil_type] / content_sum
                    continue
                self.content_dict[key] = max(0.0, self.content_dict[key] - added_value / total_soil_types)
                if self.content_dict[key] < 1e-6:
                    del self.content_dict[key]

        total_height_change = height_difference * (1 - desired_soil_relative_content_ratio * same_material_change_coeff)
        self.height_level = max(0, min(1, self.height_level + total_height_change))

        self._update_nutrition()
        self._set_tile_texture_by_soil()

        self.water.update_absorbtion_coeff()
        self.world.update_tile_height(self.in_map_position, self.height_level)

    def _set_surrounding_tiles(self, surrounding_tiles) -> None:
        """
        Caching tiles in a set of radiuses, where radius of 0 means self tile, 1 is closes neighbourhood and so on.
        combination of radiuses such as 012 means self tile, and both first and second order of the neighborhood.
        :param surrounding_tiles:
        :return:
        """
        self.surround_tiles_dict["1"] = tuple(surrounding_tiles[1])
        self.surround_tiles_dict["2"] = tuple(surrounding_tiles[2])
        self.surround_tiles_dict["01"] = tuple([self] + surrounding_tiles[1])
        self.surround_tiles_dict["12"] = tuple(surrounding_tiles[1] + surrounding_tiles[2])
        self.surround_tiles_dict["012"] = tuple([self] + surrounding_tiles[1] + surrounding_tiles[2])

    def get_surrounding_tile(self, tile_range: int | str):
        """
        :param tile_range:  if str: ["01"] - self tile + 1 radius, or ["2"] only tiles in radius of 2 (not in 1 or 0),
        if int is given returns all the tiles in a given radius + smaller radiuses, including self
        :return: 1D list of Tile instances around the tile.
        """
        if isinstance(tile_range, int):
            # yes, it is hardcoded for now
            radius_tuple = ("01", "012")
            return self.surround_tiles_dict[radius_tuple[tile_range-1]]
        return self.surround_tiles_dict[tile_range]

    def add_dead_creature(self, bodies_cnt: int, creature: Creature) -> None:
        """
                Adds a new dead body to the tile, that will be considered as meet
                :param bodies_cnt: number of new bodies
                :param creature: Creature to be used for drawing dead body
        """
        if tile_items.DeadBody.TYPE in self.all_items_dict:
            self.all_items_dict[tile_items.DeadBody.TYPE].add_dead_creature(bodies_cnt, creature)
        else:
            self._add_item(tile_items.DeadBody(self, bodies_cnt, creature))

    def _add_item(self, item: TileItem) -> None:
        """
        Adds the given item to self and, in case item is active, to the world.
        If item of the same type already exists on the tile, then the item will be re-recorded
        """
        if item.HAS_PASSIVE:
            self.passive_items_dict[item.TYPE] = item
        self.all_items_dict[item.TYPE] = item
        self.world.add_active_item(item)

    def delete_item(self, item: TileItem) -> None:
        """
        Deletes the given item from self and, in case item is active, from the world
        """
        if item.HAS_PASSIVE:
            del self.passive_items_dict[item.TYPE]
        del self.all_items_dict[item.TYPE]
        if item.HAS_PREPARE_STEP or item.HAS_STEP:
            self.world.delete_active_item(item)

    def prepare_step_tile(self):
        # --- NOTE ---
        # water and vegetation prepare operations were replace by matrix form called from terrain class

        #self.water.prepare_step_water()
        # for key in list(self.vegetation_dict.keys()):
        #     self.vegetation_dict[key].prepare_step_vegetation()
        pass

    def step_tile(self, update_visuals=False):
        if self.dead_cnt > 0:
            if random.random() < WorldProperties.body_disappear_chance:
                self.erase_dead_creature(1)  # one body is erased at a time

        # for modifier in self.modifiers:
        #     pass
        #     # water is processed automatically inside of Water class
        #     # if modifier[0] == "water source":
        #     #     self.moisture_level += modifier[1]

        #self.water.step_water(update_visuals)

        # for key in list(self.vegetation_dict.keys()):  # call via listed keys is due to potential deletion of a key
        #     self.vegetation_dict[key].step_vegetation(update_visuals)

    def draw(self, screen, pos, width, is_new_step):
        height_scale = width * 0.5  # TODO remove this value from the draw function since it is accessasble via self.world.height_scale
        #  texture_color = world_properties.soil_types[self.content_list[0][0]]["color"]   # OBSOLETE, NOW THERE IS self.texture
        # TODO optimize computations of height_pos, it must be done only once per tile and then passed as argument to
        # sub-draw functions
        height_pos = [pos[0] + self.height_level * self.world.height_direction[0] * self.world.height_scale,
                      pos[1] + self.height_level * self.world.height_direction[1] * self.world.height_scale]
        # for now drawing the highest content only
        pygame.draw.rect(screen, self.texture, (height_pos[0], height_pos[1], width, width))

        # # drawing land modifiers
        # for modifier in self.modifiers:
        #     pass
        #     # if modifier[0] == "water source":  # water source is drawn in Water class
        #     #     radius = width/2 * modifier[1] / Water.MAX_SOURCE_OUTPUT
        #     #     pygame.draw.circle(screen, (0, 0, 200, 200),
        #     #                        (height_pos[0] + width/2, height_pos[1] + width/2),
        #     #                        radius)


        self.water.draw(screen, pos, width, height_scale, height_pos, is_new_step)
        # if self.dead_creature is not None:
        #     self.dead_creature.draw(screen, pos)
        for item in self.all_items_dict.values():  # TODO draw items according to their drawing priority
            item.draw(screen, pos)

        for plant in self.vegetation_list:  # iteration is due to multiple kinds of plants
            if plant is not None:
                plant.draw(screen, pos, width, height_scale, height_pos)



