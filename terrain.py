import random
import pygame
import numpy as np
import world_properties
from copy import deepcopy

height_direction = [np.cos(np.deg2rad(210)),
                    np.sin(np.deg2rad(210))]  # with the height increase tile is move towards this vector

class Terrain:
    def __init__(self, screen, size: tuple[int, int], enable_visualization=True):
        """
        :param screen - pygame screen on which the world will be drawn
        :param size - (width, heigh) size of the generated map

        """
        self.screen = screen
        self.camera_pos = [0, 0]
        self.tile_width = 30
        self.map_size = size
        self.terrain_map = [[None for _ in range(self.map_size[1])] for _ in range(self.map_size[0])]
        self.enable_visualization = enable_visualization
        for row in range(self.map_size[0]):
            for col in range(self.map_size[1]):
                self.terrain_map[row][col] = Tile([])

    def step(self) -> None:
        for row in range(self.map_size[0]):
            for col in range(self.map_size[1]):
                surrounding_tiles = self._get_surrounding_tiles(row, col, self.terrain_map)
                self.terrain_map[row][col].prepare_step(surrounding_tiles)
        for row in range(self.map_size[0]):
            for col in range(self.map_size[1]):
                surrounding_tiles = self._get_surrounding_tiles(row, col, self.terrain_map)
                self.terrain_map[row][col].step(surrounding_tiles, self.enable_visualization)

        print(" --- Step was made ---")

    def multiple_steps(self, steps_cnt):
        enable_visualization = self.enable_visualization
        self.enable_visualization = False
        for _ in range(steps_cnt):
            self.step()
        self.enable_visualization = enable_visualization


    def _get_surrounding_tiles(self, row: int, col: int, tile_map: list[list[...]]) -> list[list[...]]:
        """
        Central tile is set to None as the surrounding around it is computed
        """
        surrounding_size = 3  # must be odd
        center = (surrounding_size - 1) // 2
        map_width, map_height = len(tile_map), len(tile_map[0])
        surrounding_tiles = [[None for _ in range(surrounding_size)] for _ in range(surrounding_size)]
        for i in range(surrounding_size):
            for j in range(surrounding_size):
                x = row + i - center
                y = col + j - center
                if x >= map_height:
                    x %= map_width
                if y >= map_width:
                    y %= map_height
                surrounding_tiles[i][j] = tile_map[x][y]
        surrounding_tiles[center][center] = None
        return surrounding_tiles

    def camera_move(self, direction: str):
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
        elif direction == "zoom out":
            self.tile_width *= 0.95
        else:
            raise "unknown camera motion direction"

    def camera_fit_view(self):
        if self.screen is None:
            return
        screen_width = self.screen.get_width()
        screen_height = self.screen.get_height()
        horizontal_size = screen_width / self.map_size[0]
        vertical_size = screen_height / self.map_size[1]
        self.tile_width = min(horizontal_size, vertical_size) * 0.98
        self.camera_pos = [0, 0]


    def draw(self):
        for row in range(self.map_size[0]):
            for col in range(self.map_size[1]):
                cur_pos = [self.camera_pos[0] + self.tile_width * row, self.camera_pos[1] + self.tile_width * col]
                self.terrain_map[row][col].draw(self.screen, cur_pos, self.tile_width)


class Water:
    # common water constants
    MOISTURE_LEVEL_TO_RISE_HEIGHT = 0.8  # if moisture reaches this level, water level starts to rise
    FLOW_RATE_TIME_CONSTANT = 0.5  # defines the speed at which water propagates into nearby tiles
    MAX_SOURCE_OUTPUT = 0.05
    MIN_SOURCE_OUTPUT = 0.001
    MOISTURE_TO_WATER_LEVEL_COEFF = 0.5  # defines the ratio at which water level increases wrt to moisture increase

    def __init__(self, self_tile):
        self.tile = self_tile  # reference to the tile that posses the water
        self.moisture_level: float = 0  # total amount of water on a tile
        self.relative_height = 0  # water starts to gain height with moisture level growing, measured from the tile height
        self.absolute_height = self.tile.height_level + self.relative_height
        self.flow_in = 0
        self.flow_out = 0
        self.moisture_lines = []
        self.source_intensity = 0

    def add_water_source(self, intensity: float):
        self.source_intensity = intensity

    def prepare_step(self, flattened_surrounding_tiles: list[...]):
        flow_out_cnt = 1
        for tile in flattened_surrounding_tiles:
            if tile.water.absolute_height - self.absolute_height < 0:
                flow_out_cnt += 1

        for tile in flattened_surrounding_tiles:
            water_diff = tile.water.absolute_height - self.absolute_height
            if water_diff < 0:
                single_tile_flow_rate_out = self.moisture_level * Water.FLOW_RATE_TIME_CONSTANT / flow_out_cnt
                tile.water.flow_in += single_tile_flow_rate_out
                self.flow_out += single_tile_flow_rate_out

    def step(self, update_visuals):
        absorption_level = 0
        for soil_type in self.tile.content_list:  # taking absorption proportionally to the soil %
            absorption_level += world_properties.soil_types[soil_type[0]]["water absorption"] * soil_type[1]

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
        elif self.relative_height < 0.000001:  # generating water lines
            if update_visuals:
                max_lines = 30
                lines_cnt = int(self.moisture_level / Water.MOISTURE_LEVEL_TO_RISE_HEIGHT * max_lines) + 3
                line_length = 0.2 + lines_cnt / max_lines * 0.4
                line_width = 1 + int( lines_cnt / max_lines * 4)
                for _ in range(lines_cnt):
                    y = random.random()
                    x_start = random.random()
                    x_end = min(x_start + line_length, 1)
                    self.moisture_lines.append([x_start, x_end, y, line_width])



        self.flow_in = 0
        self.flow_out = 0

    def draw(self, screen, pos, width, height_scale, height_pos):

        water_transparency = max(255 - self.moisture_level * 255, 20)

        water_level_offset = [pos[0] + self.absolute_height * height_direction[0] * height_scale,
                              pos[1] + self.absolute_height * height_direction[1] * height_scale]
        if self.relative_height > 0.0001:
            pygame.draw.rect(screen, (40, 40, 255, water_transparency), (water_level_offset[0], water_level_offset[1], width, width))
        elif self.moisture_level > 0.0001:
            x_min = height_pos[0]
            y_min = height_pos[1]
            for line in self.moisture_lines:
                pygame.draw.line(screen, (0, 0, 255),
                                 (x_min + line[0] * width, y_min + line[2] * width),
                                 (x_min + line[1] * width, y_min + line[2] * width), line[3])

class Tile:

    def __init__(self, surrounding_tiles: list[list[...]]):
        """
        Generates random tile
        """
        self.height_level: float = random.random()
        self.water = Water(self)
        self.content_list: list = []
        self.modifiers = []
        flattened_tiles = [item for sublist in surrounding_tiles if sublist is not None for item in sublist if item is not None]

        # purely random tile
        tile_content = {}
        for key in world_properties.soil_types:
            if key not in tile_content.keys():
                tile_content[key] = random.random()
            else:
                tile_content[key] += random.random()

        highest_content_types = sorted(tile_content, key=tile_content.get, reverse=True)[:2]  # picking two biggest values
        sum_content = 0
        for key in highest_content_types:
            sum_content += tile_content[key]
            self.content_list.append([key, tile_content[key]])
        for content in self.content_list:
            content[1] /= sum_content  # normalizing total content to [0, 1]

        for generation_property in world_properties.world_generation_properties["generation probabilities"]:
            if random.random() < generation_property[1]:
                if generation_property[0] == "water source":
                    water_source_intensity = max(random.random()*Water.MAX_SOURCE_OUTPUT,
                                                 Water.MIN_SOURCE_OUTPUT)
                    self.water.add_water_source(water_source_intensity)
                    self.modifiers.append([generation_property[0], water_source_intensity])   # second element is generating speed
                print("added ", generation_property[0])

    def prepare_step(self, surrounding_tiles: list[list[...]]):
        flattened_tiles = [item for sublist in surrounding_tiles if sublist is not None for item in sublist if
                           item is not None]

        self.water.prepare_step(flattened_tiles)

        # flow_out_cnt = 1
        # for tile in flattened_tiles:
        #     if tile.absolute_height - self.absolute_height < 0:
        #         flow_out_cnt += 1
        #
        # for tile in flattened_tiles:
        #     water_diff = tile.absolute_height - self.absolute_height
        #     if water_diff < 0:
        #         single_tile_flow_rate_out = self.moisture_level * world_properties.WATER_FLOW_RATE_TIME_CONSTANT / flow_out_cnt
        #         tile.flow_in += single_tile_flow_rate_out
        #         self.flow_out += single_tile_flow_rate_out

    def step(self, surrounding_tiles: list[list[...]], update_visuals=False):
        for modifier in self.modifiers:
            pass
            # water is processed automatically inside of Water class
            # if modifier[0] == "water source":
            #     self.moisture_level += modifier[1]
        self.water.step(update_visuals)

    def draw(self, screen, pos, width):
        height_scale = width * 0.5
        texture_color = world_properties.soil_types[self.content_list[0][0]]["color"]
        height_pos = [pos[0] + self.height_level * height_direction[0] * height_scale,
                      pos[1] + self.height_level * height_direction[1] * height_scale]
        # for now drawing the highest content only
        pygame.draw.rect(screen, texture_color, (height_pos[0], height_pos[1], width, width))

        # drawing land modifiers
        for modifier in self.modifiers:
            if modifier[0] == "water source":
                radius = width/2 * modifier[1] / Water.MAX_SOURCE_OUTPUT
                pygame.draw.circle(screen, (0, 0, 200, 200),
                                   (height_pos[0] + width/2, height_pos[1] + width/2),
                                   radius)
        self.water.draw(screen, pos, width, height_scale, height_pos)


