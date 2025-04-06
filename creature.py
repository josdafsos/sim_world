
class Creature:
    MAX_HP = 10.0
    MAX_MOVEMENT_POINTS = 10.0
    MAX_FOOD_SUPPLY = 10
    IS_CARNIVOROUS = False

    def __init__(self, self_tile):
        self.current_hp = self.MAX_HP
        self.tile = self_tile
        self.movement_points = self.MAX_MOVEMENT_POINTS
        self.current_food = self.MAX_FOOD_SUPPLY  # opposite of hunger value
        self.agent = None
        
    def eat(self):
        pass
    
    def sleep(self):
        pass
        
    def get_obs(self):
        pass
    
    def move(self):
        pass
    