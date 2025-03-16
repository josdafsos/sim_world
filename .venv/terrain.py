content_types = {
    "dirt": {
        "color": (148, 85, 49)
    },
    "rock": {
        "color": (128, 128, 128)
    },
    "sand": {
        "color": (223, 197, 94)
    }
}


class Terrain:
    def __init__(self, window, size: tuple[int, int]):
        """
        :param size - (width, heigh) size of the generated map

        """
        pass

class Tile:
    def __init__(self, surrounding_tiles: list[list[Tile | None, ...]]):
        """
        Generates random tile
        """

        self.height_level: float = None
        self.moisture_level: float = None
        self.content_list: list = None