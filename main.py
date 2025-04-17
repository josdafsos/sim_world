import time

import pygame

import keyboard_actions
import terrain
import creatures
import agents



if __name__ == '__main__':


    # Initialize pygame
    pygame.init()
    # WIDTH, HEIGHT = 700, 600
    WIDTH, HEIGHT = 1200, 800
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Sim world")

    world = terrain.Terrain(screen, (50, 50), verbose=0)  # background
    world.camera_fit_view()
    world.multiple_steps(100)

    random_cow_agent = agents.RandomCow()
    dqn_cow_agent = agents.DQNCow(verbose=1)
    # random_cow = creatures.Creature(random_cow_agent, texture="cow.png")
    #world.add_creature(creatures.Creature(random_cow_agent))
    #world.add_creature(creatures.Creature(random_cow_agent))
    world.add_creature(creatures.Creature(dqn_cow_agent, texture="cow.png", verbose=0))



    last_render_time = time.time()

    # Main loop
    running = True
    time_step_made = time.time()
    steps_made_in_a_row = 0
    while running:

        if not len(world.creatures):
            world.add_creature(creatures.Creature(dqn_cow_agent, texture="cow.png", verbose=0))  # spawning cow if everyone is dead

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keyboard_actions.parse_actions(world)
        if world.autoplay:
            world.step()

        if world.enable_visualization and time.time() - last_render_time > 0.04:
            last_render_time = time.time()
            screen.fill((0, 0, 0, 255))  # Fill background with white
            world.draw()
            pygame.display.flip()  # Update the screen

        time.sleep(2e-3)

    # Quit pygame
    pygame.quit()

