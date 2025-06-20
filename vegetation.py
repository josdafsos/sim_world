import random
import pygame
import numpy as np

class Vegetation:

    MAX_CNT = 1  # maximum number of certain vegetation_dict type per tile
    MAX_LENGTH = 1.0  # visual length of vegetation_dict as ration of tile width
    MIN_NEW_PLANT_SIZE = 0.05  # plant dies if its size below this value
    MIN_PLANT_SIZE = 0.1  # minimum size of a newly created plant
    MIN_MOISTURE_REQUIRED = 0.05  # if surrounding tile don't provide enough moisture, plants will fade
    RANDOM_DECAY_PROBABILITY = 0.005  # probability that a plant will fade a bit
    PLANTING_PROBABILITY_COEFF = 0.2  # Affects how frequently the plant will spread its seeds.
    # Original 0.2, but lower might be more realistic at a big matp

    TYPE = "base class"

    def __init__(self, self_tile):
        self.tile = self_tile
        self.vegetation_list: list[list[...]] = []  # list of vegetation_dict properties, each entity [size, x_pos, y_pos], size in (0, 1) range
        self.size = -1  # max 1
        self.count = 0

    def _create_entity(self, tile):
        return Vegetation(tile)

    def plant(self, tile, growing_power=-1.0):

        tile.world.update_vegetation_presence(tile.in_map_position, 1.0)  # Marking vegetation presence on the global map

        if self.TYPE not in tile.vegetation_dict:
            tile.vegetation_dict[self.TYPE] = self._create_entity(tile)  # TODO could a __class__ property be used here instead of this?
        if tile.vegetation_dict[self.TYPE].size < 0:
            tile.vegetation_dict[self.TYPE].size = self.MIN_NEW_PLANT_SIZE + random.random() * 0.5
        if growing_power < 0:  # growing randomly
            new_plants_cnt = min(self.MAX_CNT - tile.vegetation_dict[self.TYPE].count, 1 + random.randint(1, self.MAX_CNT))
        else:
            new_plants_cnt = min(self.MAX_CNT - tile.vegetation_dict[self.TYPE].count,
                                 1 + int(random.random() * 0.1 * growing_power))
        tile.vegetation_dict[self.TYPE].count += new_plants_cnt

        for _ in range(new_plants_cnt):
            new_plant = [random.random() * 0.7 + 0.3,  # plant size, max 1
                         random.random(),  # X coordinate
                         random.random() * 0.9 + 0.05,  # Y coordinate
                         ]
            tile.vegetation_dict[self.TYPE].vegetation_list.append(new_plant)

    def clear_dead_plants(self):
        # removing plants with size less than minimal required for life
        if self.size < self.MIN_PLANT_SIZE:
            self.count -= random.randint(1, 3)
            if self.count > 0:
                self.vegetation_list = self.vegetation_list[:self.count]  # clearing extra grass for visuals
                self.size = self.MIN_PLANT_SIZE
            else:
                del self.tile.vegetation_dict[self.TYPE]

                # removing vegetation presence flag from global map
                if not self.tile.vegetation_dict:
                    self.tile.world.update_vegetation_presence(self.tile.in_map_position, 0.0)

    def prepare_step_vegetation(self):
        total_moisture = 0
        surrounding_tiles = self.tile.surround_tiles_dict["012"]
        search_radius = 2  # radius of surrounding tiles search
        tile_x, tile_y = self.tile.in_map_position
        # for tile in surrounding_tiles:
        #     total_moisture += tile.water.moisture_level

        # TODO this part can be optimized if the required indexes are cached
        # Get wrapped row and column indices for the moisture level
        W, H = self.tile.world.map_size  # world height and width
        if (tile_x - search_radius < 0 or tile_y - search_radius < 0
                or tile_x + search_radius > W or tile_y + search_radius > H):

            row_indices = (np.arange(tile_x - search_radius, tile_x + search_radius + 1) % W)
            col_indices = (np.arange(tile_y - search_radius, tile_y + search_radius + 1) % H)
            total_moisture = np.sum(self.tile.world.pad_moisture_level_mat[np.ix_(row_indices, col_indices)])
        else:
            total_moisture = np.sum(self.tile.world.pad_moisture_level_mat[tile_x - search_radius: tile_x + search_radius + 1,
                                    tile_y - search_radius: tile_y + search_radius + 1])

        if self.tile.world.water_relative_height[tile_x, tile_y] > 1e-4:
            self.size *= 1 - 0.05 - random.random() * 0.2
        elif total_moisture < self.MIN_MOISTURE_REQUIRED or random.random() < self.RANDOM_DECAY_PROBABILITY:
            self.size *= 1 - 0.01 - random.random() * 0.05 - 0.01 * (1 - self.tile.nutrition)
        else:
            self.size *= 1 + 0.01 + random.random() * 0.1 * self.tile.nutrition
        self.size = min(self.size, 1)

        self.clear_dead_plants()

    def step_vegetation(self):
        growth_power = self.size * self.count
        surround_tiles = self.tile.surround_tiles_dict["012"]
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
        # if self.size > 1 or self.size < 1e-4:
        #     print(self.size)
        x_min = height_pos[0]
        y_min = height_pos[1]
        for plant in self.vegetation_list:
            pygame.draw.line(screen, (0, 120, 0),
                             (x_min + plant[1] * width, y_min + plant[2] * width),
                             (x_min + plant[1] * width, y_min + plant[2] * width - width * self.MAX_LENGTH * plant[0] * self.size),
                             3)

