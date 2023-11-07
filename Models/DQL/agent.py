"""
With reference to:
https://github.com/calvinfeng/machine-learning-notebook/blob/master/reinforcement_learning/deep_q_learning.py
https://calvinfeng.gitbook.io/machine-learning-notebook/unsupervised-learning/reinforcement-learning/reinforcement_learning#deep-q-learning-example 
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html 
https://www.youtube.com/watch?v=wc-FxNENg9U&t=217s 
"""
import Models.DQL.nnetwork as dqlnn
import numpy as np
import random
import torch
import torch.optim as optim
from utils import get_input_layer as input
import Models.DQL.state as State
import game.flappyNoGraphics as Game
import game.wrapped_flappy_bird as GameVisual
from collections import deque

class Agent(object):
    def __init__(self):
        """
        Porperties:
            gamma (float): Future reward discount rate.
            epsilon (float): Probability for choosing random policy.
            epsilon_decay (float): Rate at which epsilon decays toward zero.
            learning_rate (float): Learning rate for Adam optimizer.

        Returns:
            Agent
        """
        # constant parameters
        self.gamma = 0.99
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.998
        self.lr = 0.003
        self.batch_size = 64
        self.max_mem_size = 100000
        # self.input_dims = 7 * 4

        #variable parameters
        self.epsilon = 1
        self.mem_cntr = 0

        # initializing memory
        self.state_memory = np.zeros((self.max_mem_size, 4*7), dtype=np.float32)
        self.new_state_memory = np.zeros((self.max_mem_size, 4*7), dtype=np.float32)
        self.action_memory = np.zeros(self.max_mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.max_mem_size, dtype=np.float32)
        self.game_over_memory = np.zeros(self.max_mem_size, dtype=np.bool)

        #initialize networks
        self.network = dqlnn.Network(self.lr)
        # self.target_net = dqlnn.Network(self.lr)
        # self.optimizer = optim.AdamW(self.network.parameters(), lr=self.lr, amsgrad=True)

    def nextEpisode(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def getepsilon(self):
        return self.epsilon

    def remember(self, state, action, reward, next_state, game_over):
        index = self.mem_cntr % self.max_mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = next_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.game_over_memory[index] = game_over

        self.mem_cntr += 1


    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            # exploration

            # 2 in 30 = averages about 1 press every 0.5 seconds which is in the ballpark of whats required to play the game. 
            # Gives bot best start possible (as it actually has a chance of making it through the first block!)
            # in flappy bird a flap changes the gamestate a lot more than a no-flap.
            determiner = np.random.randint(0, 30);
            if (determiner <= 1):
                return 1
            return 0
        else:
            # exploitation
                state_tensor = torch.tensor([state]).to(self.network.device, dtype=torch.int32)
                action = torch.argmax(self.network.forward(state_tensor)).item()
                
        return action
    
    def updateEpsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
    def learn(self):
        if self.mem_cntr < self.batch_size:
            return
        

        self.network.optimizer.zero_grad()
        max_mem = min(self.mem_cntr, self.max_mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = torch.tensor(self.state_memory[batch]).to(self.network.device)
        new_state_batch = torch.tensor(self.new_state_memory[batch]).to(self.network.device)
        reward_batch = torch.tensor(self.reward_memory[batch]).to(self.network.device)
        action_batch = self.action_memory[batch]
        game_over_batch = torch.tensor(self.game_over_memory[batch]).to(self.network.device)

        q_current = self.network.forward(state_batch)[batch_index, action_batch]
        q_next = self.network.forward(new_state_batch)
        q_next[game_over_batch] = 0.0

        # max returns value as well as index, we only require index
        q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0]

        loss = self.network.loss(q_target, q_current).to(self.network.device)
        loss.backward()
        self.network.optimizer.step()


import keyboard
import matplotlib.pyplot as plt

def test():
    game = Game.GameState()
    agent = Agent()
    scores, eps_history = [], []
    n_games = 1000

    for i in range(n_games):
        score = 0
        game_over = False
        state_manager = State.StateManager(4)
        state = state_manager.get()
        done = False
        while not done:
            action = agent.select_action(state)
            _, reward, _ = game.frame_step(action)
            state_manager.push(game)
            next_state = state_manager.get()
            if (reward == -1):
                done = True
                final_score = score
                reward = -10
            score += reward
            agent.remember(state, action, reward, next_state, done)
            agent.learn()
            state = next_state
        agent.updateEpsilon()
        scores.append(final_score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])
        print('episode: ', i,'score: %.2f' % score,
                ' average score %.2f' % avg_score, 'epsilon %.2f' % agent.epsilon)
    plt.plot(scores)
    plt.show()



    
