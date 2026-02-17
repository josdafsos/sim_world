import multiprocessing
import os
import pickle

import neat
import creatures
from terrain import Terrain
from agents import agents


def eval_genome(genome, config):
    """
    Evaluate a single genome.
    """
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    neat_cow_agent = agents.NeatCow(net)

    # following creatures will be monitored in the world and respawn if their count is lower that the threshold
    creatures_to_respawn = (
        (creatures.Cow, neat_cow_agent, 6, {}),
    )

    world = Terrain((10, 10 ),
                    verbose=0,
                    generation_method='consistent_random',  # see other options in the description
                    steps_to_reset_world=10_000,
                    creatures_to_respawn=creatures_to_respawn,
                    )

    MAX_STEPS = 4_000  # server = 4_000
    for _ in range(MAX_STEPS):
        world.step()

    #print("Reward: ", neat_cow_agent.sum_reward)
    return neat_cow_agent.sum_reward


def train_neat_cow(checkpoint_name: str | None = None):
    """
    Run NEAT to evolve a controller for cow.
    :param checkpoint_name if given, the training will be continued from the checkpoint
    """

    # Load configuration.
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        agents.NeatCow.CONFIG_PATH,
    )

    # Create the population, which is the top-level object for a NEAT run.
    if checkpoint_name is None:
        p = neat.Population(config)
        print('---- Training of completely new population has started ----')
    else:
        p = neat.Checkpointer.restore_checkpoint(os.path.join(agents.NeatCow.SAVE_PATH, checkpoint_name))
        print('---- Training continued from checkpoint ----')

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    save_prefix = os.path.join(agents.NeatCow.SAVE_PATH, 'neat-checkpoint-' + agents.NeatCow.AGENT_NAME + '-')
    # Periodic checkpoints, similar to other examples.
    p.add_reporter(neat.Checkpointer(generation_interval=50, filename_prefix=save_prefix))

    # Use parallel evaluation across available CPU cores.
    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)

    # Run until solution or fitness threshold is reached (see config).
    winner = p.run(pe.evaluate, 5000)

    # Display the winning genome.
    print(f"\nBest genome:\n{winner!s}")

    # Save the winner for later reuse in test-feedforward.py.
    with open(os.path.join(agents.NeatCow.SAVE_PATH, agents.NeatCow.AGENT_NAME + "-winner.pickle"), "wb") as f:
        pickle.dump(winner, f)

    return winner, stats


if __name__ == '__main__':

    train_neat_cow(checkpoint_name='neat-checkpoint-NEAT_Cow-10')
