import time

import pygame

import keyboard_actions
import terrain
import creatures
import agents
import world_properties

import cProfile
import pstats

if __name__ == '__main__':

    # --- Debug flags ----
    count_execution_time = False  # if True when program is finished, highlights of longest functions sent to console


    # Initialize pygame
    pygame.init()
    # WIDTH, HEIGHT = 700, 600
    WIDTH, HEIGHT = 600, 600
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Sim world")

    world = terrain.Terrain(screen, (50, 50), verbose=0)  # background
    world.camera_fit_view()
    # world.multiple_steps(100)

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

    # Main loop
    running = True
    steps_made_in_a_row = 0
    time_step_made = time.time()
    last_render_time = time.time()

    if count_execution_time:
        profiler = cProfile.Profile()
        profiler.enable()

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
        if not has_wolf:
            world.add_creature(creatures.Wolf(memory_wolf_agent, texture="wolf_t.png", verbose=0))

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

    if count_execution_time:
        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.strip_dirs().sort_stats("cumtime").print_stats(30)  # Top 10 by cumulative time

        stats_data = stats.stats
        # --- sorted by tottime, percall sorting does not make sense as there are mostly functions called once at the top
        # custom because native function does not show all percall digits
        print(f"{'Function':<60} {'ncalls':>8} {'tottime':>12} {'percall':>12}")
        for func, (cc, nc, tt, ct, callers) in sorted(stats_data.items(), key=lambda x: x[1][2], reverse=True)[:40]:
            percall = tt / nc if nc else 0
            print(f"{func[2]} ({func[0]}:{func[1]}) {nc:>8} {tt:12.8f} {percall:12.8f}")

    # Quit pygame
    pygame.quit()

