import pygame

class Creature:
    MAX_HP: float = 10.0
    MAX_MOVEMENT_POINTS: float = 10.0
    MAX_FOOD_SUPPLY: float = 10.0
    CONSUMABLE_FOOD_TYPES: tuple[str, ...] = ("grass",)  # "fruit", "meat", "rotten meat"
    MAX_SPECIES_CNT: int = 10  # species amount the tile cannot overreach this value

    # activity that can be done towards selected tile,
    # options: move (also unify, reproduce, attack (if hostile), move onto the current tile = sleep); eat;
    # attack/split (if tile is empty, half of species go to the tile. If own tile is selected,
    # splits part of the species into random nearby location if free.
    # If the selected tile is occupied, the occupying creature is attacked even against friendly)
    AVAILABLE_ACTIONS: tuple[str, ...] = ("move", "eat")   #, "split")  # split/attack is not implemented yet

    def __init__(self, agent,
                 self_tile=None,
                 texture: str | tuple[int, int, int]=(255, 50, 50)):
        self.current_hp = self.MAX_HP
        self.tile = self_tile
        self.movement_points = self.MAX_MOVEMENT_POINTS
        self.current_food = self.MAX_FOOD_SUPPLY  # opposite of hunger value
        self.agent = agent
        self.species_cnt: int = 5  # current number os species at the tile
        self.observation = self._get_obs()
        self.texture: str | tuple[int, int, int] = texture  # can be texture or a color of a square

    def set_tile(self, tile):
        self.tile = tile

    def new_day(self):
        """ Actions executed at the beginning of a new day """
        self.movement_points = self.MAX_MOVEMENT_POINTS

    def make_action(self):

        while self.movement_points > 1e-4:
            # action tuple (tile relative number {(-1, -1), (1,0), (-1,1), etc}, action number)
            action = self.agent.predict(self.observation)
            if self.AVAILABLE_ACTIONS[action[1]] == "move":
                if action[0] == (0, 0):
                    self._sleep()
                else:
                    self._move(action[0])
            elif self.AVAILABLE_ACTIONS[action[1]] == "eat":
                self._eat(action[0])
            else:
                print(f"Warning, unknown action is called, action index: {action[1]}")

            new_obs = self._get_obs()
            self.agent.learn(self.observation, new_obs, action)
            self.observation = new_obs

    def draw(self, screen, pos, width, height_scale=None, height_pos=None):
        height_pos = [pos[0] + self.tile.height_level * self.tile.world.height_direction[0] * self.tile.world.height_scale,
                      pos[1] + self.tile.height_level * self.tile.world.height_direction[1] * self.tile.world.height_scale]

        if isinstance(self.texture, tuple):
            pygame.draw.rect(screen, self.texture, (height_pos[0], height_pos[1], 0.5*width, 0.5*width))
        else:
            print("Creatures texturing is not implemented yet")  # TODO implement textures with binary transparency

    def _get_obs(self):
        return None
        
    def _eat(self, relative_tile_pos: tuple[int, int]):
        print(f"eating at {relative_tile_pos}")
        self.movement_points -= 10.0
    
    def _sleep(self):
        print("sleeping...")
        self.movement_points -= 10.0

    def _move(self, relative_tile_pos: tuple[int, int]):
        print(f"moving to {relative_tile_pos}")
        self.movement_points -= 10.0

    