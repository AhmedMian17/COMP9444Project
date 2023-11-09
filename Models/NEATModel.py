import multiprocessing
import os.path
import pickle
import neat
from game import flappyBirdNoGraphics as GameNoGraphics
from game import wrapped_flappy_bird as Game
from utils import get_input_layer


class NEATModel:
    def __init__(self):
        configFile = os.path.join(os.path.dirname(__file__), 'NEATConfig')

        self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                  neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                  configFile)

        self.population = neat.Population(self.config)

        self.population.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        self.population.add_reporter(stats)
        self.population.add_reporter(neat.Checkpointer(20))
        self.bestGenome = None
        self.gameState = None

    def run(self, generations, checkpointFileName=""):
        if checkpointFileName != "":
            self.population = neat.Checkpointer.restore_checkpoint(checkpointFileName)
        self.bestGenome = self.population.run(self.evaluateGenomes, generations)
        with open("NEATBestGenome.pkl", "wb") as f:
            pickle.dump(self.bestGenome, f)
            f.close()

    def loadBest(self):
        with open("NEATBestGenome.pkl", "rb") as f:
            self.bestGenome = pickle.load(f)

    def playGame(self):
        self.gameState = Game.GameState()
        network = neat.nn.FeedForwardNetwork.create(self.bestGenome, self.config)
        go = True
        while go:
            networkInput = get_input_layer(self.gameState)
            networkOutput = network.activate(networkInput)[0]
            flap = networkOutput > 0.5  # sigmoid activation, output should be between 0 and 1
            _, _, terminal = self.gameState.frame_step(flap)
            if terminal:
                go = False

    def testBest(self, runs):
        self.gameState = GameNoGraphics.GameState()
        network = neat.nn.FeedForwardNetwork.create(self.bestGenome, self.config)
        fitnesses = []
        for i in range(runs):
            thisRunFitness = 0
            go = True
            while go:
                thisRunFitness += 1
                networkInput = get_input_layer(self.gameState)
                networkOutput = network.activate(networkInput)[0]
                flap = networkOutput > 0.5  # sigmoid activation, output should be between 0 and 1
                if self.gameState.frame_step(flap) or thisRunFitness > 10000:
                    go = False
                    fitnesses.append(thisRunFitness)
        print(fitnesses)

    @staticmethod
    def evaluateGenomes(genomes, config):
        gameState = GameNoGraphics.GameState()
        for genome_id, genome in genomes:
            network = neat.nn.FeedForwardNetwork.create(genome, config)
            runs = 10
            averageFitness = 0
            for i in range(runs):
                thisRunFitness = 0
                go = True
                while go:
                    thisRunFitness += 1
                    networkInput = get_input_layer(gameState)
                    networkOutput = network.activate(networkInput)[0]
                    flap = networkOutput > 0.5  # sigmoid activation, output should be between 0 and 1
                    if gameState.frame_step(flap) or thisRunFitness > 10000:
                        go = False
                        averageFitness += thisRunFitness / runs
            genome.fitness = averageFitness

# Multithreaded, works very slowly since it has to initialise a new game each time
# def evaluateGenome(genomeId, genome, config, gameState):
#     genome.fitness = 0.0
#     network = neat.nn.FeedForwardNetwork.create(genome, config)
#     go = True
#     while go:
#         genome.fitness += 1
#         networkInput = get_input_layer(gameState)
#         networkOutput = network.activate(networkInput)[0]
#         flap = networkOutput > 0.5  # sigmoid activation, output should be between 0 and 1
#         if gameState.frame_step(flap) or genome.fitness > 1000:
#             go = False
#
#
# def evaluateProcess(gameState, genomes, config):
#     for genomeId, genome in genomes:
#         evaluateGenome(genomeId, genome, config, gameState)
#
#
# def evaluateGenomesParallel(genomes, config):
#     numberOfProcesses = 10
#     gameStateList = [Game.GameState() for _ in range(numberOfProcesses)]
#     genomesPerThread = round(len(genomes) / numberOfProcesses)  # make sure this is divisible!
#     relevantGenomes = [genomes[i * genomesPerThread:(i + 1) * genomesPerThread] for i in range(0, numberOfProcesses)]
#     processes = []
#
#     for genomeList, gameState in zip(relevantGenomes, gameStateList):
#         process = multiprocessing.Process(target=evaluateProcess, args=([gameState, genomeList, config]))
#         processes.append(process)
#         process.start()
#
#     for process in processes:
#         process.join()
