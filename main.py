import time

import keyboard_actions
import terrain
import creatures
from agents import agents
from graphics import graphics

import cProfile
import pstats
import pygame
import _world_logic


# main priority TODO list
# 1. !!! TODO there is a bug that dupes creatures, could be related to the split action
# 3. multiprocessing for simulation and faster agent training
# 5. Optimize rendering for big maps (implement render accuracy levels) and refactor rendering
# 4. Roads made by creatures frequently walking through same tiles, progress depends on creature mass
# 6. Hint for controls and probably controls re-factoring
# ? - should vegetation be implemented as TileItems?
# ? - Try to compile with Cython computationally intensive functions


def computation_benchmarking():
    start_time = time.time()
    ITERATIONS_CNT = 20
    STEPS_PER_SIM = 10_000

    random_cow_agent = agents.RandomCow()
    for _ in range(ITERATIONS_CNT):
        creatures_to_respawn = (
            (creatures.Cow, random_cow_agent, 6, {}),
        )
        world = terrain.Terrain((20, 20),
                                verbose=0,
                                generation_method='consistent_random',  # see other options in the description
                                steps_to_reset_world=10_000,
                                creatures_to_respawn=creatures_to_respawn,
                                )
        for __ in range(STEPS_PER_SIM):
            world.step()

    steps_per_s = STEPS_PER_SIM * ITERATIONS_CNT / (time.time() - start_time)
    print(f"Simulation speed is: {steps_per_s} steps / second")

def game_run():
    screen = graphics.get_screen()
    random_cow_agent = agents.RandomCow()
    dqn_cow_agent = agents.DQNCow(verbose=0,
                                  epsilon=(1.0, 0.075, int(2e6)),
                                  agent_version="new_agent")
    neat_cow_agent = agents.NeatCow(model_name='neat-checkpoint-NEAT_Cow-10')
    # memory_wolf_agent = agents.DQNMemoryWolf(verbose=1)

    # following creatures will be monitored in the world and respawn if their count is lower that the threshold
    creatures_to_respawn = (
        # (creatures.Cow, random_cow_agent, 6),
        (creatures.Cow, neat_cow_agent, 6, {'verbose': 2}),
        # (creatures.Cow, neat_cow_agent, 6, {}),
        # (creatures.Wolf, memory_wolf_agent, 6),
    )

    world = terrain.Terrain((20, 20),
                            screen=screen,
                            verbose=0,
                            generation_method='consistent_random',  # see other options in the description
                            steps_to_reset_world=10_000,
                            creatures_to_respawn=creatures_to_respawn,
                            )
    world.camera_fit_view()
    # world.multiple_steps(100)

    # world.add_creature(creatures.Cow(neat_cow_agent, texture="cow_t.png", verbose=2))
    # world.add_creature(creatures.Cow(cow_agent, texture="cow_t.png", verbose=0))
    # world.add_creature(creatures.Wolf(memory_wolf_agent, texture="wolf_t.png", verbose=0))

    # Main loop
    running = True
    # steps_made_in_a_row = 0
    time_step_made = time.time()
    last_render_time = time.time()

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
    # Quit pygame
    pygame.quit()

if __name__ == '__main__':

    # --- Debug flags ----
    count_execution_time = True  # if True when program is finished, highlights of longest functions sent to console

    if count_execution_time:
        print("Debug option enabled: count_execution_time")
        profiler = cProfile.Profile()
        profiler.enable()

    game_run()
    #computation_benchmarking()

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



