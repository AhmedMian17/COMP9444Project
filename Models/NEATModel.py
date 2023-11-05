import os.path
import neat
from game import flappyBirdNoGraphics as Game
from utils import get_gamestate_info_tensor

class NEATModel:
    def __init__(self):
        configFile = os.path.join(os.path.dirname(__file__), 'NEATConfig')

        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             configFile)

        self.population = neat.Population(config)

        self.population.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        self.population.add_reporter(stats)
        self.population.add_reporter(neat.Checkpointer(10))

    def run(self, generations):
        self.population.run(self.evaluateGenomes, generations)

    @staticmethod
    def evaluateGenomes(genomes, config):
        gameState = Game.GameState()

        for genome_id, genome in genomes:
            genome.fitness = 0.0
            network = neat.nn.FeedForwardNetwork.create(genome, config)
            go = True
            while go:
                genome.fitness += 1
                networkInput = get_gamestate_info_tensor(gameState)
                networkOutput = network.activate(networkInput)[0]
                flap = networkOutput > 0.5  # sigmoid activation, output should be between 0 and 1
                if gameState.frame_step(flap):
                    go = False

