import random
import pygame

class Vegetation:
    MAX_CNT = 1  # maximum number of certain vegetation type per tile
    MAX_LENGTH = 1.0  # visual length of vegetation as ration of tile width
    MIN_NEW_PLANT_SIZE = 0.05 # plant dies if its size below this value
    MIN_PLANT_SIZE = 0.1  # minimum size of a newly created plant
    MIN_MOISTURE_REQUIRED = 0.05  # if surrounding tile don't provide enough moisture, plants will fade
    RANDOM_DECAY_PROBABILITY = 0.01  # probability that a plant will fade a bit

    def __init__(self, self_tile):
        self.tile = self_tile
        self.vegetation_list: list[list[...]] = []  # list of vegetation properties, each entity [size, x_pos, y_pos], size in (0, 1) range
        self.type = "base class"

    def _create_entity(self, tile):
        return Vegetation(tile)

    def plant(self, tile, growing_power=-1.0):
        if not self.type in tile.vegetation:
            tile.vegetation[self.type] = self._create_entity(tile)

        if growing_power < 0:  # growing randomly
            new_plants_cnt = min(self.MAX_CNT - len(tile.vegetation[self.type].vegetation_list),  + random.randint(1, self.MAX_CNT))
            for _ in range(new_plants_cnt):
                new_plant = [random.random() * (1 - self.MIN_NEW_PLANT_SIZE) + self.MIN_NEW_PLANT_SIZE + 1e-4,  # plant size, max 1
                             random.random(),  # X coordinate
                             random.random() * 0.9 + 0.05,  # Y coordinate
                            ]
                tile.vegetation[self.type].vegetation_list.append(new_plant)
        else:
            new_plants_cnt = min(self.MAX_CNT - len(tile.vegetation[self.type].vegetation_list),
                                 1 + int(random.random() * 0.1 * growing_power))
            for _ in range(new_plants_cnt):
                new_plant = [random.random() * 0.05 + self.MIN_NEW_PLANT_SIZE + 1e-4,
                             # plant size, max 1
                             random.random(),  # X coordinate
                             random.random() * 0.9 + 0.05,  # Y coordinate
                             ]
                tile.vegetation[self.type].vegetation_list.append(new_plant)

    def clear_dead_plants(self):
        # removing plants with size less than minimal required for life
        self.vegetation_list = [plant for plant in self.vegetation_list if plant[0] > self.MIN_PLANT_SIZE]

    def prepare_step(self, flattened_surrounding_tiles: list[...]):
        pass

    def step(self, update_visuals, flattened_surrounding_tiles: list[...]):
        pass

    def draw(self, screen, pos, width, height_scale, height_pos):
        pass

class Grass(Vegetation):
    MAX_LENGTH = 0.3
    MAX_CNT = 10


    def __init__(self, self_tile):
        super().__init__(self_tile)
        self.type = "grass"

    def _create_entity(self, tile):
        return Grass(tile)

    def prepare_step(self, flattened_surrounding_tiles: list[...]):
        total_moisture = 0
        for tile in flattened_surrounding_tiles:
            total_moisture += tile.water.moisture_level

        for plant in self.vegetation_list:
            if total_moisture < self.MIN_MOISTURE_REQUIRED or random.random() < self.RANDOM_DECAY_PROBABILITY:
                plant[0] *= 1 - 0.01 - random.random() * 0.05 - 0.01 * (1 - self.tile.nutrition)
            else:
                plant[0] *= min(1 + 0.01 + random.random() * 0.1 * self.tile.nutrition, 1)


    def step(self, update_visuals, flattened_surrounding_tiles: list[...]):
        self.clear_dead_plants()
        growth_power = 0
        for plant in self.vegetation_list:
            if plant[0] > 0.5:
                growth_power += plant[0]
        planting_probability = 0.2 * growth_power / (self.MAX_CNT * len(flattened_surrounding_tiles))
        for tile in flattened_surrounding_tiles:
            if planting_probability > random.random():
                self.plant(tile)


    def draw(self, screen, pos, width, height_scale, height_pos):
        x_min = height_pos[0]
        y_min = height_pos[1]
        for plant in self.vegetation_list:
            pygame.draw.line(screen, (0, 120, 0),
                             (x_min + plant[1] * width, y_min + plant[2] * width),
                             (x_min + plant[1] * width, y_min + plant[2] * width - width * self.MAX_LENGTH * plant[0]),
                             3)

