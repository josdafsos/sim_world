import time

import keyboard_actions
import terrain
import creatures
import agents
from graphics import graphics

import cProfile
import pstats
import pygame


if __name__ == '__main__':

    # --- Debug flags ----
    count_execution_time = False  # if True when program is finished, highlights of longest functions sent to console

    if count_execution_time:
        print("Debug option enabled: count_execution_time")

    screen = graphics.get_screen()
    # random_cow_agent = agents.RandomCow()
    dqn_cow_agent = agents.DQNCow(verbose=1,
                                  epsilon=(1.0, 0.05, int(1e7)),
                                  agent_version="new_agent")
    memory_wolf_agent = agents.DQNMemoryWolf(verbose=1)
    cow_agent = dqn_cow_agent

    # following creatures will be monitored in the world and respawn if their count is lower that the threshold
    creatures_to_respawn = (
        (creatures.Cow, dqn_cow_agent, 6),
        (creatures.Wolf, memory_wolf_agent, 6),
    )

    world = terrain.Terrain(screen,
                            (20, 20),
                            verbose=0,
                            generation_method='consistent_random',  # see other options in the description
                            steps_to_reset_world=100_000,
                            creatures_to_respawn=creatures_to_respawn,
                            )
    world.camera_fit_view()
    # world.multiple_steps(100)
    # world.add_creature(creatures.Cow(cow_agent, texture="cow_t.png", verbose=0))
    # world.add_creature(creatures.Wolf(memory_wolf_agent, texture="wolf_t.png", verbose=0))

    # Main loop
    running = True
    # steps_made_in_a_row = 0
    time_step_made = time.time()
    last_render_time = time.time()

    if count_execution_time:
        profiler = cProfile.Profile()
        profiler.enable()

    while running:

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN or event.type == pygame.KEYUP:
                keyboard_actions.parse_actions(world,
                                               event.key,
                                               event.type == pygame.KEYDOWN)

        if world.autoplay:
            world.step()

        if world.enable_visualization and time.time() - last_render_time > 0.05:
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

