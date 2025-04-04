import time

import pygame
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
    # Main loop
    running = True
    time_step_made = time.time()
    steps_made_in_a_row = 0
    while running:

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:  # Check if a key is pressed
                if event.key == pygame.K_w:  # Check if it's the spacebar
                    world.move_camera("up")
                elif event.key == pygame.K_s:
                    world.move_camera("down")
                elif event.key == pygame.K_a:
                    world.move_camera("left")
                elif event.key == pygame.K_d:
                    world.move_camera("right")
                elif event.key == pygame.K_x:
                    world.move_camera("zoom in")
                elif event.key == pygame.K_z:
                    world.move_camera("zoom out")
                elif event.key == pygame.K_SPACE:
                    pass

        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            if time.time() - time_step_made > 0.250 / (steps_made_in_a_row + 1):
                world.step()
                time_step_made = time.time()
                steps_made_in_a_row = min(steps_made_in_a_row + 1, 5000)
        elif steps_made_in_a_row > 0:
            steps_made_in_a_row = 0.0

        # screen.fill((0, 0, 0, 255))  # Fill background with white
        pygame.draw.rect(background, (0, 0, 0, 255), (0, 0, WIDTH, HEIGHT))  # clearing the screen, fill does not work due to transparency
        world.draw()
        screen.blit(background, (0, 0))
        pygame.display.flip()  # Update the screen

    # Quit pygame
    pygame.quit()

