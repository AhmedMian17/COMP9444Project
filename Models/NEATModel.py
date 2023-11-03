import os.path
import neat


class NEATModel:
    def __init__(self):
        configFile = os.path.join(os.path.dirname(__file__), 'NEATConfig')

        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             configFile)

        population = neat.Population(config)

        population.run(self.evaluateGenomes, 300)

    @staticmethod
    def evaluateGenomes(genomes, config):
        for genome_id, genome in genomes:
            genome.fitness = 4.0
            net = neat.nn.FeedForwardNetwork.create(genome, config)
