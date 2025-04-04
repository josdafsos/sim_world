import random
import pygame
import numpy as np
import world_properties
from copy import deepcopy

height_direction = [np.cos(np.deg2rad(210)),
                    np.sin(np.deg2rad(210))]  # with the height increase tile is move towards this vector

class Terrain:
    def __init__(self, screen, size: tuple[int, int]):
        """
        :param screen - pygame screen on which the world will be drawn
        :param size - (width, heigh) size of the generated map

        """
        self.screen = screen
        self.camera_pos = [0, 0]
        self.tile_width = 30
        self.map_size = size
        self.terrain_map = [[None for _ in range(self.map_size[1])] for _ in range(self.map_size[0])]
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
                self.terrain_map[row][col].step(surrounding_tiles)

        print(" --- Step was made ---")

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

    def move_camera(self, direction: str):
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

    def draw(self):
        for row in range(self.map_size[0]):
            for col in range(self.map_size[1]):
                cur_pos = [self.camera_pos[0] + self.tile_width * row, self.camera_pos[1] + self.tile_width * col]
                self.terrain_map[row][col].draw(self.screen, cur_pos, self.tile_width)


class Water:
    # TODO pack all water-related things into a separate class
    # TODO probably the water problem is just with the visualization
    def __init__(self):
        pass


class Tile:

    def __init__(self, surrounding_tiles: list[list[...]], enable_visualization=True):
        """
        Generates random tile
        """

        self.height_level: float = random.random()
        self.moisture_level: float = 0  # total amount of water on a tile
        self.water_relative_height = 0  # water starts to gain height with moisture level growing, measured from the tile height
        self.water_absolute_height = self.height_level + self.water_relative_height
        self.water_flow_in = 0
        self.water_flow_out = 0
        self.water_moisture_lines = []
        self.water_enable_visualization = enable_visualization
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
                    self.modifiers.append([generation_property[0],
                                           max(random.random()*world_properties.WATER_MAX_SOURCE_OUTPUT,
                                               world_properties.WATER_MIN_SOURCE_OUTPUT)])   # second element is generating speed
                print("added ", generation_property[0])

    def prepare_step(self, surrounding_tiles: list[list[...]]):
        flattened_tiles = [item for sublist in surrounding_tiles if sublist is not None for item in sublist if
                           item is not None]

        flow_out_cnt = 1
        for tile in flattened_tiles:
            if tile.water_absolute_height - self.water_absolute_height < 0:
                flow_out_cnt += 1

        for tile in flattened_tiles:
            water_diff = tile.water_absolute_height - self.water_absolute_height
            if water_diff < 0:
                single_tile_flow_rate_out = self.moisture_level * world_properties.WATER_FLOW_RATE_TIME_CONSTANT / flow_out_cnt
                tile.water_flow_in += single_tile_flow_rate_out
                self.water_flow_out += single_tile_flow_rate_out

    def step(self, surrounding_tiles: list[list[...]]):
        for modifier in self.modifiers:
            if modifier[0] == "water source":
                self.moisture_level += modifier[1]

        # water physics
        # water absorption
        absorption_level = 0
        for soil_type in self.content_list:  # taking absorption proportionally to the soil %
            absorption_level += world_properties.soil_types[soil_type[0]]["water absorption"] * soil_type[1]

        flattened_tiles = [item for sublist in surrounding_tiles if sublist is not None for item in sublist if
                           item is not None]
        self.moisture_level += self.water_flow_in - self.water_flow_out
        self.moisture_level -= absorption_level
        self.water_moisture_lines = []  # for visual effects of moisture
        if self.moisture_level < 0.0001:
            self.moisture_level = 0
        elif self.water_relative_height < 0.0001:  # generating water lines
            if self.water_enable_visualization:
                lines_cnt = int(self.moisture_level / world_properties.WATER_MOISTURE_LEVEL_TO_RISE_HEIGHT * 50)
                line_length = 0.2
                for _ in range(lines_cnt):
                    y = random.random()
                    x_start = random.random()
                    x_end = min(x_start + line_length, 1)
                    self.water_moisture_lines.append([x_start, x_end, y])

        # water starts to fill the tile only when a threshold is reached
        self.water_relative_height = max(self.moisture_level - world_properties.WATER_MOISTURE_LEVEL_TO_RISE_HEIGHT, 0)
        self.water_absolute_height = self.height_level + self.water_relative_height

        self.water_flow_in = 0
        self.water_flow_out = 0

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
                pygame.draw.circle(screen, (0, 0, 200, 200),
                                   (height_pos[0] + width/2, height_pos[1] + width/2),
                                   width/2 * modifier[1])
        # drawing water level
        water_transparency = max(255 - self.moisture_level * 255, 20)

        water_level_offset = [pos[0] + self.water_absolute_height * height_direction[0] * height_scale,
                              pos[1] + self.water_absolute_height * height_direction[1] * height_scale]
        if self.water_relative_height > 0.0001:
            pygame.draw.rect(screen, (40, 40, 255, water_transparency), (water_level_offset[0], water_level_offset[1], width, width))
        elif self.moisture_level > 0.0001:
            x_min = height_pos[0]
            y_min = height_pos[1]
            for line in self.water_moisture_lines:
                pygame.draw.line(screen, (0, 0, 255),
                                 (x_min + line[0] * width, y_min + line[2] * width),
                                 (x_min + line[1] * width, y_min + line[2] * width), 2)

