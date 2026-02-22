import random
import pygame
import numpy as np
from numba import njit

import utils
from graphics import graphics
import _world_logic

# IDEA evolution of vegetation if a plant is put into certain conditions
# i.e. grass can evolve into lili plant if submerged into water

@njit
def fast_sum(A, rows, cols):
    s = 0.0
    for i in range(len(rows)):
        s += A[rows[i], cols[i]]
    return s

class Vegetation:
    """
    Base vegetation class, any vegetation class must inherit it.
    """
    MAX_CNT: int = 1  # maximum number of certain vegetation_dict type per tile
    MAX_LENGTH: float = 1.0  # visual length of vegetation_dict as ration of tile width
    MIN_NEW_PLANT_SIZE: float = 0.05  # plant dies if its size below this value
    MIN_PLANT_SIZE = 0.1  # minimum size of a newly created plant
    MIN_MOISTURE_REQUIRED: float = 0.05  # if surrounding tile don't provide enough moisture, plants will fade
    MOISTURE_SEARCH_RADIUS: int = 2  # defines the radius of the surrounding tile to compute moisture for the plant
    RANDOM_DECAY_PROBABILITY: float = 0.005  # probability that a plant will fade a bit
    PLANTING_PROBABILITY_COEFF: float = 0.2  # Affects how frequently the plant will spread its seeds.
    # Original 0.2, but lower might be more realistic at a big matp
    MASS: float = 1.0  # mass of a single plant (assumed in kg, kinda). Used in interactions with vegetation. # TODO implement

    TYPE: str = "base class"  # type is a unique identifier of a class
    GROUP: str = "base class group"  # group is an identifier of a collection of vegetation types, such as
    # various types of grass or trees
    PLANT_INDEX_MAPPING: int = utils.plant_group_index_mapping_dict[GROUP]

    def __init__(self,
                 self_tile,
                 must_be_planted: bool = True,
                 texture: str | tuple[int, int, int] = (255, 255, 255)):
        """
        Places vegetation on the corresponding Tile if that is possible.
        After initialization, interact with vegetation only via corresponding tile instance.
        :param self_tile: Tile on which vegetation instance is spawned
        :param must_be_planted: do not use it, it is an internal parameter for correct initialization
        :param texture: string, name of a texture file. Default None - rendered with primitives if implemented.
        """

        if self._check_same_group_exist(self_tile):  # block growing of vegetation of the same group in a tile
            self.texture = None  # some of the attributes are attempted to be access by child class
            # and thus they must be initialized even on failure to make a new plant on a tile
            return
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
        self.moisture_row_indices = (np.arange(tile_x - self.MOISTURE_SEARCH_RADIUS, tile_x + self.MOISTURE_SEARCH_RADIUS + 1) % W)
        self.moisture_col_indices = (np.arange(tile_y - self.MOISTURE_SEARCH_RADIUS, tile_y + self.MOISTURE_SEARCH_RADIUS + 1) % H)
        # mat of tile indexes to compute surrounding moisture level
        self.moisutre_tile_idx = np.ix_(self.moisture_row_indices, self.moisture_col_indices)

        self.moisture_tile_mask = np.zeros(self.tile.world.map_size, dtype=bool)
        self.moisture_tile_mask[self.moisutre_tile_idx] = True

        self.moisture_buffer_value: float = 0.0  # stores moisture value computed during previous step

        # --- visual properties ---
        if isinstance(texture, str):
            self.texture = graphics.get_texture(texture)
        else:
            self.texture = texture  # a default drawing method will be used if exist
        self.scaled_textures = []  # since there might be multiple plants the scaled textures is a list
        self.on_rescale()

        if must_be_planted:
            self.plant(self.tile)

    def _init_visual_vegetation(self):
        self.visual_vegetation_list = tuple((random.random() * 0.7 + 0.3,  # plant size, max 1
                                             random.random(),  # X coordinate, as a proportion of tile width
                                             random.random() * 0.9 + 0.05)  # Y coordinate, as a proportion of tile width
                                            for _ in range(self.MAX_CNT))

    def _evolve(self) -> None:
        """
        Plant can evolve into another one if certain conditions are met.
        The function is called at the end of vegetation step function
        """
        pass

    def _check_same_group_exist(self, tile) -> bool:
        """
        Function checks if a vegetation of the same GROUP as self exists on a given tile
        :param tile: tile to verify
        :return: True if a plant of the same group already exists on a tile, otherwise returns False
        """
        return tile.vegetation_list[self.PLANT_INDEX_MAPPING] is not None  # TODO uncomment
        # for plant in tile.vegetation_dict.values():
        #     if plant.GROUP == self.GROUP and plant.TYPE != self.TYPE:
        #         return True
        # return False

    def plant(self, tile, growing_power=-1.0) -> None:
        """
        Function to create a new plant on a tile
        :param tile: Tile to plant a plant
        :param growing_power:
        """
        tile.world.update_vegetation_presence(tile.in_map_position, 1.0)  # Marking vegetation presence on the global map

        if self._check_same_group_exist(tile):
            if self.TYPE != tile.vegetation_list[self.PLANT_INDEX_MAPPING].TYPE:  # blocking growth of same vegetation group on the same tile
                return
        else:
             # creating a new instance if it did not exist
            #tile.vegetation_dict[self.TYPE] = self.__class__(tile, texture=self.texture, must_be_planted=False)
            tile.vegetation_list[self.PLANT_INDEX_MAPPING] = self.__class__(tile, texture=self.texture, must_be_planted=False)

        # before refactoring
        # if self.TYPE not in tile.vegetation_dict:  # creating a new instance if it did not exist
        #     if self._check_same_group_exist(tile):  # blocking growth of same vegetation group on the same tile
        #         return
        #     tile.vegetation_dict[self.TYPE] = self.__class__(tile, texture=self.texture, must_be_planted=False)

        #vegetation = tile.vegetation_dict[self.TYPE]
        vegetation = tile.vegetation_list[self.PLANT_INDEX_MAPPING]

        if vegetation.size < 0:
            vegetation.size = self.MIN_NEW_PLANT_SIZE + random.random() * 0.5
        if growing_power < 0:  # growing randomly
            new_plants_cnt = min(self.MAX_CNT - vegetation.count, 1 + random.randint(1, self.MAX_CNT))
        else:
            new_plants_cnt = min(self.MAX_CNT - vegetation.count,
                                 1 + int(random.random() * 0.1 * growing_power))
        vegetation.count += new_plants_cnt

    def _delete_plant(self):  # TODO this function should not be protected as it is used outside
        """
        Deletion of the plant instance from corresponding tile
        """
        self.tile.vegetation_list[self.PLANT_INDEX_MAPPING] = None
        for plant in self.tile.vegetation_list:
            if plant is not None:
                return
        self.tile.world.update_vegetation_presence(self.tile.in_map_position, 0.0)
        # if self.TYPE in self.tile.vegetation_dict:  # for some reason during evolution an error emerges here
        #     del self.tile.vegetation_dict[self.TYPE]
        # if not self.tile.vegetation_dict:  # checking if the tile is completely free of vegetation
        #     self.tile.world.update_vegetation_presence(self.tile.in_map_position, 0.0)

        # TODO refactor vegetation_presence matrix to INT values, add 1 when a new plant is created,
        #  and subtract one when a plant dies. It will be faster than looping though palnts

    def clear_dead_plants(self):
        # removing plants with size less than minimal required for life
        if self.size < self.MIN_PLANT_SIZE:
            self.count -= random.randint(1, 3)
            if self.count > 0:
                self.size = self.MIN_PLANT_SIZE
            else:
                self._delete_plant()

    def _compute_plant_growth(self):
        # TODO quite time-consuming function, same for cactus growth function
        """
        Function to compute growth/decay of plant on a tile during simulation step.
        Can be redefined to obtain other plant growing behaviour.
        """

        tile_x, tile_y = self.tile.in_map_position
        #total_moisture = np.sum(self.tile.world.pad_moisture_level_mat[self.moisutre_tile_idx])  # old inefficient computation
        #total_moisture = self.tile.world.moisture_level_mat[self.moisture_tile_mask].sum()  # still could be somehow optimized
        total_moisture =  fast_sum(self.tile.world.pad_moisture_level_mat,
                                   self.moisture_row_indices,
                                   self.moisture_col_indices)
        self.moisture_buffer_value = total_moisture

        # C call implementation
        self.size = _world_logic.compute_plant_growth_grass(self.size,
                                                self.tile.world.water_relative_height[tile_x, tile_y],
                                                total_moisture,
                                                self.MIN_MOISTURE_REQUIRED,
                                                self.RANDOM_DECAY_PROBABILITY,
                                                self.tile.nutrition
                                                )

        # # old function implementation
        # if self.tile.world.water_relative_height[tile_x, tile_y] > 1e-4:  # plant is dying due to high water level
        #     self.size *= 1 - 0.05 - random.random() * 0.2
        # elif total_moisture < self.MIN_MOISTURE_REQUIRED or random.random() < self.RANDOM_DECAY_PROBABILITY:
        #     self.size *= 1 - 0.01 - random.random() * 0.05 - 0.01 * (1 - self.tile.nutrition)
        # else:
        #     self.size *= 1 + 0.01 + random.random() * 0.1 * self.tile.nutrition
        # self.size = min(self.size, 1)

    def prepare_step_vegetation(self):
        self._compute_plant_growth()
        self.clear_dead_plants()

    def step_vegetation(self):
        growth_power = self.size * self.count
        surround_tiles = self.tile.surround_tiles_dict["012"]
        # TODO too many random calls, the function can be optimized if amount of random calls is reduced
        planting_probability = self.PLANTING_PROBABILITY_COEFF * growth_power / (self.MAX_CNT * len(surround_tiles))
        for tile in surround_tiles:
            if (tile.world.water_relative_height[tile.in_map_position] < 1e-5 and
                    planting_probability > random.random()):  # no planting in water, old: tile.water.relative_height < 1e-5
                self.plant(tile)

        self._evolve()

    def on_rescale(self):
        """ This function is called if the width of Tile has changed"""
        # NOTE: this function could somehow be moved to graphics
        if not isinstance(self.texture, tuple) and self.tile.world is not None:
            self.scaled_textures = []
            width = self.tile.world.tile_width
            for plant_idx in range(self.count):
                plant = self.visual_vegetation_list[plant_idx]
                scaled_size = width * self.MAX_LENGTH * plant[0] * self.size
                self.scaled_textures.append(pygame.transform.scale(self.texture,
                                                                   (scaled_size, scaled_size)))  # previously 0.5*...

    def draw(self, screen, pos, width, height_scale, height_pos):
        pass


class Cactus(Vegetation):
    # TODO add effect of damaging anyone who try to eat it or to pass by

    MAX_LENGTH = 0.8
    MAX_CNT = 3
    MASS = 20
    TYPE = "cactus"
    GROUP = "grass"
    PLANT_INDEX_MAPPING = utils.plant_group_index_mapping_dict[GROUP]
    MAX_MOISTURE_REQUIRED: float = 1e-3  # cactus starts to die if there is too much moisture around
    PLANTING_PROBABILITY_COEFF = 0.025

    def __init__(self, self_tile, texture="cactus_t.png", **kwargs):
        super().__init__(self_tile,
                         texture=texture,  # texture="cow_t.png",
                         **kwargs,
                         )
        if not isinstance(self.texture, tuple):
            self.MAX_LENGTH = 2  # scaling size if texture exists

    def _has_neighbouring_cactus(self) -> bool:
        """
        :return: True if in a radius of one another cactus is presented
        """
        for tile in self.tile.get_surrounding_tile("1"):
            if tile.has_vegetation([self.TYPE]):
                return True
        return False

    def _compute_plant_growth(self):
        """
        Function to compute growth/decay of plant on a tile during simulation step.
        Can be redefined to obtain other plant growing behaviour.
        """
        # TODO this is a time consuming function that should be optimized
        tile_x, tile_y = self.tile.in_map_position
        # total_moisture = self.tile.world.moisture_level_mat[
        #     self.moisture_tile_mask].sum()  # still could be somehow optimized
        total_moisture =  fast_sum(self.tile.world.pad_moisture_level_mat,
                                   self.moisture_row_indices,
                                   self.moisture_col_indices)
        self.moisture_buffer_value = total_moisture

        has_neighbour_cactus = self._has_neighbouring_cactus()  # neighbouring cactuses kill each other

        if self.tile.world.water_relative_height[tile_x, tile_y] > 1e-4:  # plant is dying due to high water level
            self.size *= 1 - 0.05 - random.random() * 0.2
        elif (total_moisture > self.MAX_MOISTURE_REQUIRED or
              random.random() < self.RANDOM_DECAY_PROBABILITY):
            self.size *= 1 - 0.01 - random.random() * 0.05
        elif has_neighbour_cactus:
            self.size *= 1 - 0.0005 - random.random() * 0.0005
        else:
            self.size *= 1 + 0.001 + random.random() * 0.005
        self.size = min(self.size, 1)

    def plant(self, tile, growing_power=-1.0) -> None:
        """
        Cactus cannot be planted on anything but sand
        """
        if tile.highest_content_type != "sand" or self._has_neighbouring_cactus():
            return
        super().plant(tile, growing_power)

    def _init_visual_vegetation(self):
        self.visual_vegetation_list = tuple((random.random() * 0.7 + 0.3,  # plant size, max 1
                                             random.random() * 0.3,  # X coordinate, as a proportion of tile width
                                             random.random() * 0.3 + 0.5)  # Y coordinate, as a proportion of tile width
                                            for _ in range(self.MAX_CNT))

    def on_rescale(self):
        """ This function is called if the width of Tile has changed"""
        # NOTE: this function could somehow be moved to graphics
        if not isinstance(self.texture, tuple) and self.tile.world is not None:
            self.scaled_textures = []
            width = self.tile.world.tile_width
            for plant_idx in range(self.count):
                plant = self.visual_vegetation_list[plant_idx]
                scaled_size = width * self.MAX_LENGTH * plant[0] * self.size
                self.scaled_textures.append(pygame.transform.scale(self.texture,
                                                                   (scaled_size/2, scaled_size)))  # previously 0.5*...

    def draw(self, screen, pos, width, height_scale, height_pos):
        x_min = height_pos[0]
        y_min = height_pos[1]
        if isinstance(self.texture, tuple):
            for plant_idx in range(self.count):
                if plant_idx >= len(self.visual_vegetation_list):
                    print(f"Trying to draw non-existing cactus, total count {self.count},"
                          f" visual vegetation length {len(self.visual_vegetation_list)}")
                plant = self.visual_vegetation_list[plant_idx]
                pygame.draw.line(screen, self.texture,
                                 (x_min + plant[1] * width, y_min + plant[2] * width),
                                 (x_min + plant[1] * width, y_min + plant[2] * width - width * self.MAX_LENGTH * plant[0] * self.size),
                                 3)
        else:
            self.on_rescale()  # must be called all the time to update actual texture size. Can it be optimized?
            for plant_idx in range(self.count):
                plant = self.visual_vegetation_list[plant_idx]
                self.tile.world.screen.blit(self.scaled_textures[plant_idx],
                                   (x_min + plant[1] * width,
                                    y_min + plant[2] * width - self.scaled_textures[plant_idx].get_rect().height))

            # highlight cactus position:
            # pygame.draw.circle(screen, (255, 0, 0), (x_min + width/2, y_min + width/2), 5)


class Grass(Vegetation):
    MAX_LENGTH = 0.3
    MAX_CNT = 10
    MASS = 10
    TYPE = "grass"
    GROUP = "grass"
    PLANT_INDEX_MAPPING = utils.plant_group_index_mapping_dict[GROUP]

    def __init__(self, self_tile, texture=(0, 120, 0), **kwargs):
        super().__init__(self_tile,
                         texture=texture,  # texture="cow_t.png",
                         **kwargs
                         )

    def _evolve(self) -> None:
        """
        Grass can evolve into: cactus
        """

        # checking catus evlotion. It can grow only in dry conditions and send
        if self.moisture_buffer_value < 1e-3 and self.tile.highest_content_type == 'sand':
            if random.random() < 0.05:  # some percent chance to evolve into cactus
                self._delete_plant()
                Cactus(self.tile)

    def draw(self, screen, pos, width, height_scale, height_pos):
        x_min = height_pos[0]
        y_min = height_pos[1]
        if isinstance(self.texture, tuple):
            for plant_idx in range(self.count):
                plant = self.visual_vegetation_list[plant_idx]
                pygame.draw.line(screen, self.texture,
                                 (x_min + plant[1] * width, y_min + plant[2] * width),
                                 (x_min + plant[1] * width, y_min + plant[2] * width - width * self.MAX_LENGTH * plant[0] * self.size),
                                 3)
        else:
            if len(self.scaled_textures) != self.count:
                self.on_rescale()
            for plant_idx in range(self.count):
                plant = self.visual_vegetation_list[plant_idx]
                self.tile.world.screen.blit(self.scaled_textures[plant_idx],
                                   (x_min + plant[1] * width, y_min + plant[2] * width))

