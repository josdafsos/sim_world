import random
import pygame
import numpy as np

class Vegetation:

    MAX_CNT = 1  # maximum number of certain vegetation_dict type per tile
    MAX_LENGTH = 1.0  # visual length of vegetation_dict as ration of tile width
    MIN_NEW_PLANT_SIZE = 0.05  # plant dies if its size below this value
    MIN_PLANT_SIZE = 0.1  # minimum size of a newly created plant
    MIN_MOISTURE_REQUIRED = 0.05  # if surrounding tile don't provide enough moisture, plants will fade
    MOISTURE_SEARCH_RADIUS = 2  # defines the radius of the surrounding tile to compute moisture for the plant
    RANDOM_DECAY_PROBABILITY = 0.005  # probability that a plant will fade a bit
    PLANTING_PROBABILITY_COEFF = 0.2  # Affects how frequently the plant will spread its seeds.
    # Original 0.2, but lower might be more realistic at a big matp

    TYPE = "base class"

    def __init__(self, self_tile):
        self.tile = self_tile
        # tuple of vegetation_dict visual properties, each entity [size, x_pos, y_pos], size in (0, 1) range
        self.visual_vegetation_list: tuple[tuple[...]] = []
        self._init_visual_vegetation()
        self.size = -1  # max 1, size of all plants of this class on this tile
        self.count = 0

        # defining indexes of surrounding tiles to compute moisture
        W, H = self.tile.world.map_size
        tile_x, tile_y = self.tile.in_map_position
        # right now only wrapped world is considered.
        # To consider other topologies compliment row and col variable definition
        row_indices = (np.arange(tile_x - self.MOISTURE_SEARCH_RADIUS, tile_x + self.MOISTURE_SEARCH_RADIUS + 1) % W)
        col_indices = (np.arange(tile_y - self.MOISTURE_SEARCH_RADIUS, tile_y + self.MOISTURE_SEARCH_RADIUS + 1) % H)
        # mat of tile indexes to compute surrounding moisture level
        self.moisutre_tile_idx = np.ix_(row_indices, col_indices)

    def _init_visual_vegetation(self):
        self.visual_vegetation_list = tuple((random.random() * 0.7 + 0.3,  # plant size, max 1
                                             random.random(),  # X coordinate, as a proportion of tile width
                                             random.random() * 0.9 + 0.05)  # Y coordinate, as a proportion of tile width
                                            for _ in range(self.MAX_CNT))

    def plant(self, tile, growing_power=-1.0) -> None:
        """
        Function to create a new plant on a tile
        :param tile: Tile to plant a plant
        :param growing_power:
        """
        tile.world.update_vegetation_presence(tile.in_map_position, 1.0)  # Marking vegetation presence on the global map

        if self.TYPE not in tile.vegetation_dict:  # creating a new instance if it did not exist
            tile.vegetation_dict[self.TYPE] = self.__class__(tile)
        if tile.vegetation_dict[self.TYPE].size < 0:
            tile.vegetation_dict[self.TYPE].size = self.MIN_NEW_PLANT_SIZE + random.random() * 0.5
        if growing_power < 0:  # growing randomly
            new_plants_cnt = min(self.MAX_CNT - tile.vegetation_dict[self.TYPE].count, 1 + random.randint(1, self.MAX_CNT))
        else:
            new_plants_cnt = min(self.MAX_CNT - tile.vegetation_dict[self.TYPE].count,
                                 1 + int(random.random() * 0.1 * growing_power))
        tile.vegetation_dict[self.TYPE].count += new_plants_cnt

    def clear_dead_plants(self):
        # removing plants with size less than minimal required for life
        if self.size < self.MIN_PLANT_SIZE:
            self.count -= random.randint(1, 3)
            if self.count > 0:
                self.size = self.MIN_PLANT_SIZE
            else:
                del self.tile.vegetation_dict[self.TYPE]

                # removing vegetation presence flag from global map
                if not self.tile.vegetation_dict:
                    self.tile.world.update_vegetation_presence(self.tile.in_map_position, 0.0)

    def _compute_plant_growth(self):
        """
        Function to compute growth/decay of plant on a tile during simulation step.
        Can be redefined to obtain other plant growing behaviour.
        """

        tile_x, tile_y = self.tile.in_map_position
        total_moisture = np.sum(self.tile.world.pad_moisture_level_mat[self.moisutre_tile_idx])

        if self.tile.world.water_relative_height[tile_x, tile_y] > 1e-4:  # plant is dying due to high water level
            self.size *= 1 - 0.05 - random.random() * 0.2
        elif total_moisture < self.MIN_MOISTURE_REQUIRED or random.random() < self.RANDOM_DECAY_PROBABILITY:
            self.size *= 1 - 0.01 - random.random() * 0.05 - 0.01 * (1 - self.tile.nutrition)
        else:
            self.size *= 1 + 0.01 + random.random() * 0.1 * self.tile.nutrition
        self.size = min(self.size, 1)

    def prepare_step_vegetation(self):

        self._compute_plant_growth()
        self.clear_dead_plants()

    def step_vegetation(self):
        growth_power = self.size * self.count
        surround_tiles = self.tile.surround_tiles_dict["012"]
        # TODO too many random calls, the function can be optimized if amount of random calls is reduced
        planting_probability = self.PLANTING_PROBABILITY_COEFF * growth_power / (self.MAX_CNT * len(surround_tiles))
        for tile in surround_tiles:
            if (planting_probability > random.random() and
                    tile.world.water_relative_height[tile.in_map_position] < 1e-5):  # no planting in water, old: tile.water.relative_height < 1e-5
                self.plant(tile)

    def draw(self, screen, pos, width, height_scale, height_pos):
        pass

class Grass(Vegetation):
    MAX_LENGTH = 0.3
    MAX_CNT = 10
    TYPE = "grass"

    def __init__(self, self_tile):
        super().__init__(self_tile)

    def _create_entity(self, tile):
        return Grass(tile)

    def draw(self, screen, pos, width, height_scale, height_pos):
        x_min = height_pos[0]
        y_min = height_pos[1]
        for plant_idx in range(self.count):
            plant = self.visual_vegetation_list[plant_idx]
            pygame.draw.line(screen, (0, 120, 0),
                             (x_min + plant[1] * width, y_min + plant[2] * width),
                             (x_min + plant[1] * width, y_min + plant[2] * width - width * self.MAX_LENGTH * plant[0] * self.size),
                             3)


