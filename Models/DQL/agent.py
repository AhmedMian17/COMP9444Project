"""
With reference to:
https://github.com/calvinfeng/machine-learning-notebook/blob/master/reinforcement_learning/deep_q_learning.py
https://calvinfeng.gitbook.io/machine-learning-notebook/unsupervised-learning/reinforcement-learning/reinforcement_learning#deep-q-learning-example 
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html 
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

MAX_MEMORY = 10000

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
        self.gamma = 0.95
        self.epsilon = 0.99
        self.epsilon_min = 0.00
        self.epsilon_decay = 0.996
        self.learning_rate = 5e-2
        self.batch_size = 32

        self.memory = deque([], maxlen=MAX_MEMORY)
        self.policy_net = dqlnn.Network()
        self.target_net = dqlnn.Network()
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.learning_rate, amsgrad=True)

    def nextEpisode(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def getepsilon(self):
        return self.epsilon

    def remember(self, state, action, reward, next_state, game_over):
        if (len(self.memory) >= MAX_MEMORY - 20):
            for _ in range(2000):
                self.memory.pop()
        self.memory.append((state, action, reward, next_state, game_over))


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
            with torch.no_grad():
                return_value = self.policy_net(state)
                if (np.argmax(return_value) >= 2 or np.argmax(return_value) < 0):
                    raise Exception("unexpected argmax on return value agent.py 39")
            return np.argmax(return_value)
        
    def replay(self):
        try:
            memory_batch = random.sample(self.memory, self.batch_size)
        except:
            #do nothing if not enough samples in memory
            return
        
        for state, action, reward, next_state, game_over in memory_batch:
            target = reward
            if not game_over:
                target = reward + self.gamma * torch.max(self.policy_net(next_state))
            target_f = self.policy_net(state)
            criterion = torch.nn.SmoothL1Loss()
            loss = criterion(target_f, torch.tensor(target))

            #optimize model
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
            self.optimizer.step()

import keyboard
import matplotlib.pyplot as plt

def test():
    # initialise game and agent
    agent = Agent()
    state = State.StateFrame(4)
    episode_number = 0
    episode_max = 1500

    # initialise params for memory tracking
    previous_state = torch.tensor(0)
    curr_state = torch.tensor(0)

    # def remember(self, state, action, reward, next_state, game_over):
    reward_record = []

    #for each episode
    while episode_number <= episode_max:
        if (episode_number > 1498):
            game = GameVisual.GameState()
        else:
            game = Game.GameState()
        reward = 0
        game_over = 0
        episode_reward = 0
        while (game_over == 0):
            
            # select an actiona
            action = agent.select_action(state.push(game))
            _, reward, _ = game.frame_step(action)

            # shuffle state around for memory
            if (curr_state.shape == state.get().shape):
                previous_state = curr_state
            else:
                previous_state = state.get()
            curr_state = state.get()

            # amplify punishment for losing
            if (reward == -1):
                record = episode_reward
                reward = -1 - episode_reward
                game_over = 1

            # add reward.
            episode_reward += reward
            
            # add to memory
            agent.remember(previous_state, action, reward, curr_state, game_over)

            #admin stuff
            if keyboard.is_pressed("q"):
                return
        agent.replay()    
        episode_number += 1
        agent.nextEpisode()
        reward_record.append(record)
        if (episode_number % 100 == 0):
            epsilon = agent.getepsilon()
            print("episode: ", episode_number, "reward: ", record, "epsilon: ", epsilon)
    
    #plot out reward record against episode number
    plt.plot(reward_record)
    while keyboard.is_pressed("q") == False:
        plt.show()
    return

    
