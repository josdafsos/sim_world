import time

import pygame

import keyboard_actions
import terrain
import creatures
import agents
import world_properties

if __name__ == '__main__':


    # Initialize pygame
    pygame.init()
    # WIDTH, HEIGHT = 700, 600
    WIDTH, HEIGHT = 600, 600
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Sim world")

    world = terrain.Terrain(screen, (50, 50), verbose=0)  # background
    world.camera_fit_view()
    world.multiple_steps(100)

    random_cow_agent = agents.RandomCow()
    dqn_cow_agent = agents.DQNCow(verbose=1, agent_version="new_agent")
    memory_wolf_agent = agents.DQNMemoryWolf(verbose=1)
    # random_cow = creatures.Creature(random_cow_agent, texture="cow.png")
    cow_agent = dqn_cow_agent
    # dqn_wolf_agent = agents.DQNWolf(verbose=1)
    #world.add_creature(creatures.Creature(random_cow_agent))
    #world.add_creature(creatures.Creature(random_cow_agent))

    world.add_creature(creatures.Cow(cow_agent, texture="cow_t.png", verbose=0))
    world.add_creature(creatures.Wolf(memory_wolf_agent, texture="wolf_t.png", verbose=0))



    last_render_time = time.time()

    # Main loop
    running = True
    time_step_made = time.time()
    steps_made_in_a_row = 0
    while running:

        # if len(world.creatures) < 4:
        cnt = 0
        has_wolf = False
        for creature in world.creatures:
            cnt += creature.species_cnt
            has_wolf = has_wolf or creature.CREATURE_ID == world_properties.WOLF_ID
        if cnt <= 5:
            world.add_creature(creatures.Cow(cow_agent, texture="cow_t.png", verbose=0))
            # world.add_creature(creatures.Wolf(dqn_wolf_agent, texture="wolf_t.png", verbose=0))
        #if not has_wolf:
        #    world.add_creature(creatures.Wolf(dqn_wolf_agent, texture="wolf_t.png", verbose=0))

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

        # time.sleep(2e-3)

    # Quit pygame
    pygame.quit()

