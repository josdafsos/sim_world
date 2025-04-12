import random
import time

import pygame
import numpy as np
import world_properties
import vegetation
import creatures
import agents


class Terrain:
    height_direction = (np.cos(np.deg2rad(210)),
                        np.sin(np.deg2rad(210)))  # with the height increase tile is move towards this vector
    HOURS_PER_STEP = 6  # how many hours are passed after each mini-step is made

    def __init__(self, screen, size: tuple[int, int], enable_visualization=True,
                 enable_map_loop_visualization=False,
                 verbose: int = 0,
                 is_round_map=True):
        """
        :param screen - pygame screen on which the world will be drawn
        :param size - (width, heigh) size of the generated map
        :param verbose - {0, 1, 2} sets the amount of information printed in the console
        :param is_round_map If true the map connects opposite ends of the map

        """

        # --- visual settings ---
        self.screen = screen
        self.camera_pos = [0, 0]
        self.camera_visible_x_range = []  # coordinates of start and end of visible tiles along x directions
        self.camera_visible_y_range = []  # coordinates of start and end of visible tiles along y directions
        self.camera_visible_tiles_cnt = 0  # number of tiles visible on the screen at the moment
        self.tile_width = 30
        self.height_scale = self.tile_width * 0.5
        self.enable_visualization = enable_visualization
        self.font = pygame.font.SysFont(None, 48)

        # --- map settings ---
        self.is_round_map = is_round_map
        self.enable_map_loop_visualization = enable_map_loop_visualization and is_round_map
        self.map_size = size
        self.terrain_map: list[list[Tile | ...]] = [[None for _ in range(self.map_size[1])] for _ in range(self.map_size[0])]

        # --- other settings ---
        self.verbose = verbose
        self.total_steps = 0
        self.autoplay = False
        self.current_time_hours: int = 0  # current time of the day
        self.creatures = []  # list of all creatures currently living in the world

        for row in range(self.map_size[0]):
            for col in range(self.map_size[1]):
                self.terrain_map[row][col] = Tile(self, [], (row, col))
        for row in range(self.map_size[0]):
            for col in range(self.map_size[1]):
                _, distance_sorted_tiles = self._get_surrounding_tiles(row, col, self.terrain_map, search_diameter=5)
                self.terrain_map[row][col]._set_surrounding_tiles(distance_sorted_tiles)

        self.terrain_map = tuple(tuple(inner) for inner in self.terrain_map)
        self.camera_move("update")

    def add_creature(self, creature: creatures.Creature, position: None | tuple[int, int] = None) -> None:
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

    def step(self) -> None:
        time_before_step = time.time()

        if self.creatures and self.current_time_hours <= 24 - self.HOURS_PER_STEP:  # in case we have any creatures, iterate through them
            for creature in self.creatures:
                creature.make_action()
            self.current_time_hours += self.HOURS_PER_STEP
        else:
            self.current_time_hours = 0
            for row in range(self.map_size[0]):
                for col in range(self.map_size[1]):
                    self.terrain_map[row][col].prepare_step()
            for row in range(self.map_size[0]):
                for col in range(self.map_size[1]):
                    self.terrain_map[row][col].step(self.enable_visualization)  #(surrounding_tiles, self.enable_visualization)
            i = 0
            while i < len(self.creatures):
                is_rotten = self.creatures[i].new_day()
                if is_rotten:
                    self.creatures.pop(i)
                else:
                    i += 1

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

    def get_creature_on_tile(self, tile):
        """ Returns a creature on the tile. If tile is empty returns None"""
        for creature in self.creatures:
            if creature.tile == tile:
                return creature
        return None

    def delete_creature(self, creature):
        if creature in self.creatures:
            self.creatures.remove(creature)
            if self.verbose > 0:
                print(f"creature deleted, total creature count: {len(self.creatures)}")

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

    def camera_move(self, direction: str, is_camera_rescaled = False):
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

        self.height_scale = self.tile_width * 0.5
        if is_camera_rescaled:
            if self.tile_width < 30:
                self.font = pygame.font.SysFont(None, 30)
            else:
                self.font = pygame.font.SysFont(None, 48)

            for creature in self.creatures:
                creature.on_rescale()  # changing texture sizes

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
                    self.terrain_map[row][col].draw(self.screen, cur_pos, self.tile_width)
                    tiles_drawn += 1
            for creature in self.creatures:
                row, col = creature.tile.in_map_position
                cur_pos = [self.camera_pos[0] + self.tile_width * row, self.camera_pos[1] + self.tile_width * col]
                creature.draw(self.screen, cur_pos, self.tile_width)

        else:
            print("camera loop visualization is not yet implemented")  # TODO


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

    def prepare_step(self):
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

    def step(self, update_visuals):
        absorption_level = 0
        for soil_type in self.tile.content_list:  # taking absorption proportionally to the soil %
            absorption_level += world_properties.soil_types[soil_type[0]]["water absorption"] * soil_type[1]
        if self.tile.vegetation_dict:  # presence of vegetation_dict reduces water  absorption
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



        self.flow_in = 0.0
        self.flow_out = 0.0

    def draw(self, screen, pos, width, height_scale, height_pos):

        water_transparency = max(255 - self.moisture_level * 255, 20)

        water_level_offset = [pos[0] + self.absolute_height * self.tile.world.height_direction[0] * height_scale,
                              pos[1] + self.absolute_height * self.tile.world.height_direction[1] * height_scale]
        if self.relative_height > 0.0001:
            pygame.draw.rect(screen, (40, 40, 255, water_transparency), (water_level_offset[0], water_level_offset[1], width, width))
        elif self.tile.world.camera_visible_tiles_cnt < 3000:
            if self.source_intensity > 1e-4:
                radius = width/2 * self.source_intensity / self.MAX_SOURCE_OUTPUT
                pygame.draw.circle(screen, (0, 0, 200, 200),
                                   (height_pos[0] + width/2, height_pos[1] + width/2),
                                   radius)

            if self.moisture_level > 0.0001: #width > 8:  # second condition is for drawing optimization
                x_min = height_pos[0]
                y_min = height_pos[1]
                for line in self.moisture_lines:
                    pygame.draw.line(screen, (0, 0, 255),
                                     (x_min + line[0] * width, y_min + line[2] * width),
                                     (x_min + line[1] * width, y_min + line[2] * width), line[3])


class Tile:

    def __init__(self, world, surrounding_tiles: list[list[...]], in_map_position):
        """
        Generates random tile
        """
        self.height_level: float = random.random()
        self.world = world
        self.water = Water(self)
        self.content_list: list = []
        self.modifiers = []
        self.vegetation_dict = {}
        self.surround_tiles_dict = {}
        self.nutrition = 0
        self.in_map_position: tuple[int, int] = in_map_position   # (row, col) indexes of the tile in the world


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
            self.nutrition += content[1] * world_properties.soil_types[content[0]]["nutritional value"]

        for generation_property in world_properties.world_generation_properties["generation probabilities"]:
            if random.random() < generation_property[1]:
                if generation_property[0] == "water source":
                    water_source_intensity = max(random.random()*Water.MAX_SOURCE_OUTPUT,
                                                 Water.MIN_SOURCE_OUTPUT)
                    self.water.add_water_source(water_source_intensity)
                    self.modifiers.append([generation_property[0], water_source_intensity])   # second element is generating speed
                elif generation_property[0] == "grass":
                    if not "grass" in self.vegetation_dict:
                        self.vegetation_dict["grass"] = vegetation.Grass(self)

                    self.vegetation_dict["grass"].plant(self)

                print("added ", generation_property[0])

    def has_vegetation(self, vegetation_list: tuple[str, ...]):
        has_vegetation = False
        existing_plant_key = None
        for plant in vegetation_list:
            if plant in self.vegetation_dict:
                has_vegetation = True
                existing_plant_key = plant
                break
        return has_vegetation, existing_plant_key

    def consume_vegetation(self, vegetation_options) -> bool:
        has_vegetation, plant_key = self.has_vegetation(vegetation_options)
        if not has_vegetation:
            return False
        del self.vegetation_dict[plant_key]
        return True

    def _set_surrounding_tiles(self, surrounding_tiles):
        self.surround_tiles_dict["1"] = tuple(surrounding_tiles[1])
        self.surround_tiles_dict["2"] = tuple(surrounding_tiles[2])
        self.surround_tiles_dict["01"] = tuple([self] + surrounding_tiles[1])
        self.surround_tiles_dict["12"] = tuple(surrounding_tiles[1] + surrounding_tiles[2])
        self.surround_tiles_dict["012"] = tuple([self] + surrounding_tiles[1] + surrounding_tiles[2])

    def get_surrounding_tile(self, tile_range: int | str):
        if isinstance(tile_range, int):
            # yes, it is hardcoded for now
            radius_tuple = ("01", "012")
            return self.surround_tiles_dict[radius_tuple[tile_range]]
        return self.surround_tiles_dict[tile_range]

    def prepare_step(self):
        self.water.prepare_step()
        for key in list(self.vegetation_dict.keys()):
            self.vegetation_dict[key].prepare_step()

    def step(self, update_visuals=False):
        for modifier in self.modifiers:
            pass
            # water is processed automatically inside of Water class
            # if modifier[0] == "water source":
            #     self.moisture_level += modifier[1]
        self.water.step(update_visuals)

        for key in list(self.vegetation_dict.keys()):  # call via listed keys is due to potential deletion of a key
            self.vegetation_dict[key].step(update_visuals)


    def draw(self, screen, pos, width):
        height_scale = width * 0.5  # TODO remove this value from the draw function since it is accessasble via self.world.height_scale
        texture_color = world_properties.soil_types[self.content_list[0][0]]["color"]
        height_pos = [pos[0] + self.height_level * self.world.height_direction[0] * self.world.height_scale,
                      pos[1] + self.height_level * self.world.height_direction[1] * self.world.height_scale]
        # for now drawing the highest content only
        pygame.draw.rect(screen, texture_color, (height_pos[0], height_pos[1], width, width))

        # drawing land modifiers
        for modifier in self.modifiers:
            pass
            # if modifier[0] == "water source":  # water source is drawn in Water class
            #     radius = width/2 * modifier[1] / Water.MAX_SOURCE_OUTPUT
            #     pygame.draw.circle(screen, (0, 0, 200, 200),
            #                        (height_pos[0] + width/2, height_pos[1] + width/2),
            #                        radius)


        self.water.draw(screen, pos, width, height_scale, height_pos)
        for plant in self.vegetation_dict.values():  # iteration is due to multiple kinds of plants
            plant.draw(screen, pos, width, height_scale, height_pos)



