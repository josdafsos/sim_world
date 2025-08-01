"""

Definition of every action that a creature can make (as a Class inherited from Action class.

"""


def action_attack(attacking_creature, other_creature):
    own_attack = attacking_creature.species_cnt * attacking_creature.SINGLE_CREATURE_STRENGTH
    other_attack = other_creature.species_cnt * other_creature.SINGLE_CREATURE_STRENGTH
    min_damage = 0.1  # for some reason an attack can deal a negative damage, idk why
    own_attack = max(min_damage, own_attack)
    other_attack = max(min_damage, other_attack)

    attacking_creature.apply_damage(other_attack)
    other_creature.apply_damage(own_attack)
    if attacking_creature.verbose > 0:
        print(
            f"Creature {attacking_creature}, cnt={attacking_creature.species_cnt} attacks \n {other_creature}, cnt={other_creature.species_cnt} "
            f"corresponding damage {own_attack} and {other_attack}")

    attacking_creature.consume_food(0.5)
    other_creature.consume_food(0.25)
    attacking_creature.movement_points -= 1.0


class Action:
    ACTION_SPACE_SIZE: int = -1  # size of input action set. Example, a walking creature can move to nearby tiles, 8 directions
    # (options) in total; a creature can go to sleep (only one option)
    TYPE: str = "base action"  # name of the action class

    def __init__(self, creature):
        self.creature = creature

    def make_action(self, action_number: int, has_done_actions: bool, **kwargs) -> bool:
        """
        The action must return True if that is the last action on its turn, for example if it sleeps.
        Otherwise, returns False
        :param action_number: integer to specify the exact action (such as direction of action)
        :param has_done_actions: must be True if the creature has done an action previously during same turn
        :return: bool: is last action?
        """
        pass


class Sleep(Action):
    ACTION_SPACE_SIZE = 1
    TYPE = "sleep"

    def __init__(self, creature):
        super().__init__(creature)

    def make_action(self, action_number, has_done_actions, **kwargs) -> bool:
        if has_done_actions:
            self.creature.consume_food(0.005, enable_heal=False)  # small penalty for making restricted action
            return
        self.creature.consume_food(0.05, enable_heal=True)
        self.creature.movement_points = self.creature.MAX_MOVEMENT_POINTS
        return True


class Eat(Action):
    """ Eat from the tile that a creature is standing on"""

    ACTION_SPACE_SIZE = 1
    TYPE = "eat"

    def __init__(self, creature):
        super().__init__(creature)

    def make_action(self, action_number, has_done_actions, **kwargs) -> bool:

        self.creature.movement_points -= 1.0
        has_eaten_food = self.creature.tile.eat_from_tile(self.creature.CONSUMABLE_FOOD_TYPES)

        if self.creature.verbose > 0:
            print(f"eating at the current tile, success = {has_eaten_food}")
        if has_eaten_food:
            self.creature.current_food = min(self.creature.current_food + 5.0, self.creature.MAX_FOOD_SUPPLY)
        else:
            self.creature.consume_food(0.05)

        return False


class EatAround(Action):
    """same as eat, but also allows to eat from surrounding tiles without movement to them"""
    ACTION_SPACE_SIZE = 9
    TYPE = "eat_around"
    TILE_POS_ACTION_MAPPING = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1))

    def __init__(self, creature):
        super().__init__(creature)

    def make_action(self, action_number, has_done_actions, **kwargs) -> bool:

        relative_tile_pos = self.TILE_POS_ACTION_MAPPING[action_number]
        if relative_tile_pos == (0, 0):
            self.creature.movement_points -= 1.0
        else:
            self.creature.movement_points -= 2.0
        tile_to_eat = self.creature.tile.world.get_tile_by_index(self.creature.tile.in_map_position, relative_tile_pos)
        has_eaten_food = tile_to_eat.eat_from_tile(self.creature.CONSUMABLE_FOOD_TYPES)

        if self.creature.verbose > 0:
            print(f"eating around at {relative_tile_pos}, success = {has_eaten_food}")
        if has_eaten_food:
            self.creature.current_food = min(self.creature.current_food + 5.0, self.creature.MAX_FOOD_SUPPLY)
        else:
            self.creature.consume_food(0.05)
        return False


class Move(Action):
    ACTION_SPACE_SIZE = 8
    TYPE = "move"
    TILE_POS_ACTION_MAPPING = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))

    def __init__(self, creature):
        super().__init__(creature)

    def make_action(self, action_number, has_done_actions, **kwargs) -> bool:
        """
        The action must return True if that is the last action on its turn, for example if it sleeps.
        Otherwise, returns False
        :return: bool: is last action?
        """
        relative_tile_pos = self.TILE_POS_ACTION_MAPPING[action_number]
        if self.creature.verbose > 0:
            print(f"moving to {relative_tile_pos}")
        # new_row = self.tile.in_map_position[0] + relative_tile_pos[0]
        # new_col = self.tile.in_map_position[1] + relative_tile_pos[1]
        new_tile = self.creature.world.get_tile_by_index(self.creature.tile.in_map_position, relative_tile_pos)
        # required_movement = self._get_movement_difficulty(new_tile)
        required_movement = self.creature.get_movement_difficulty(self.creature.tile, new_tile)
        if self.creature.movement_points < required_movement * 0.5:  # attempt to make illegal move
            self.creature.consume_food(0.01)  # small penalty
            return True

        other_creature = self.creature.world.get_creature_on_tile(new_tile)
        if other_creature is None:
            self.creature.tile = new_tile  # don't use set_tile function here because it will update obs twice and do unnecessary rescale
            self.creature.movement_points = max(0.0, self.creature.movement_points - required_movement)
            self.creature.consume_food(required_movement / 20.0)
        else:

            if other_creature.CREATURE_ID == self.creature.CREATURE_ID:
                if self.creature.species_cnt == self.creature.MAX_SPECIES_CNT or other_creature == self.creature.MAX_SPECIES_CNT:
                    self.creature.consume_food(0.005)
                    self.creature.movement_points -= 0.1
                if self.creature.species_cnt + other_creature.species_cnt <= self.creature.MAX_SPECIES_CNT:  # creatures fully merge
                    other_creature.species_cnt += self.creature.species_cnt
                    self.creature.movement_points = 0.0  # to block any further action
                    self.creature.world.remove_creature(self)
                else:
                    remaining_species = self.creature.species_cnt + other_creature.species_cnt - self.creature.MAX_SPECIES_CNT
                    other_creature.species_cnt = self.creature.MAX_SPECIES_CNT
                    self.creature.species_cnt = remaining_species
                    self.creature.consume_food(0.05)
                    self.creature.movement_points -= 0.5
            else:
                action_attack(self.creature, other_creature)  # attack already consumes food and movement

        return False


class Split(Action):
    ACTION_SPACE_SIZE = 8
    TYPE = "split"
    TILE_POS_ACTION_MAPPING = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))

    def __init__(self, creature):
        super().__init__(creature)

    def make_action(self, action_number, has_done_actions, **kwargs) -> bool:
        """ If tile is empty splits half of the creature on the new tile. Otherwise attacks ANY creature on the tile"""

        relative_tile_pos = self.TILE_POS_ACTION_MAPPING[action_number]
        new_tile = self.creature.world.get_tile_by_index(self.creature.tile.in_map_position, relative_tile_pos)
        other_creature = self.creature.world.get_creature_on_tile(new_tile)
        # if self is other_creature:  # checking if the creature attacks itself
        #     if self.verbose > 0:
        #         print("suicide attempt (creature is trying to attack itself")
        #     other_creature = None

        if other_creature is None:
            if self.creature.species_cnt > 1:
                self.creature.current_food -= 0.05  # subtracted from both creatures
                self.creature.movement_points -= 1.0

                new_creature = self.creature.__class__(self.creature._agent,
                                                       self.creature.tile,
                                                       texture=self.creature.texture,
                                                       verbose=self.creature.verbose,
                                                       creature_to_copy=self.creature)

                remaining_species_cnt = self.creature.species_cnt // 2
                moved_species_cnt = self.creature.species_cnt - remaining_species_cnt
                new_creature.set_tile(new_tile)
                new_creature.species_cnt = moved_species_cnt
                self.creature.species_cnt = remaining_species_cnt
                new_creature_position = (self.creature.tile.in_map_position[0] + relative_tile_pos[0],
                                         self.creature.tile.in_map_position[1] + relative_tile_pos[1])
                self.creature.world.add_creature(new_creature, new_creature_position)
            else:
                self.creature.current_food -= 0.05
                self.creature.movement_points -= 0.1
        else:
            action_attack(self.creature, other_creature)  # attack already consumes food and movement

        return False

