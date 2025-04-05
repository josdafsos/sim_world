import time

import pygame

import keyboard_actions
import terrain

if __name__ == '__main__':
    # Initialize pygame
    pygame.init()

    # Set up the display
    WIDTH, HEIGHT = 700, 700
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Sim world")

    background = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

    # Define colors
    BLUE = (0, 0, 255)

    # Define square properties
    square_size = 50
    square_x, square_y = (WIDTH - square_size) // 2, (HEIGHT - square_size) // 2  # Centered

    world = terrain.Terrain(background, (50, 50))
    world.camera_fit_view()

    # Main loop
    running = True
    time_step_made = time.time()
    steps_made_in_a_row = 0
    while running:

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False


        keyboard_actions.parse_actions(world)

        # screen.fill((0, 0, 0, 255))  # Fill background with white
        pygame.draw.rect(background, (0, 0, 0, 255), (0, 0, WIDTH, HEIGHT))  # clearing the screen, fill does not work due to transparency
        world.draw()
        screen.blit(background, (0, 0))
        pygame.display.flip()  # Update the screen

    # Quit pygame
    pygame.quit()

