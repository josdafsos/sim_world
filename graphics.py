"""
Module to implement simple graphics operations

"""

import pygame


class Graphics:
    """
    Note must be reworked as singleton in future
    """

    def __init__(self):
        # --- Initialize pygame ---
        pygame.init()
        WIDTH, HEIGHT = 600, 600
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Sim world")

        # --- Other ---
        self.textures = {}


    def get_screen(self) -> pygame.Surface:
        """
        :return: screen instance on which visualization happens
        """
        return self.screen

    def get_texture(self, texture_path: str):
        """
        Returns pygame texture if such file exist,
        otherwise returns tuple(255, 255, 255) used as an error texture
        :param texture_path - string with name of a texture file with its extension
        """
        if texture_path not in self.textures:
            try:
                self.textures[texture_path] = pygame.image.load("textures//" + texture_path).convert_alpha()
            except:
                print(F"texture file {texture_path} could not be found")
                self.textures[texture_path] = (255, 255, 255)
        return self.textures[texture_path]


graphics = Graphics()  # rework as singleton to avoid this instance

