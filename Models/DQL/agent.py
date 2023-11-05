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

MAX_MEMORY = 2000

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
        self.epsilon = 0.95
        self.epsilon_min = 0.005
        self.epsilon_decay = 0.999
        self.learning_rate = 5e-3
        self.batch_size = 32

        self.policy_net = dqlnn.Network()
        self.target_net = dqlnn.Network()
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.learning_rate, amsgrad=True)

    def remember(self, state, action, reward, next_state, game_over):
        self.memory.append((state, action, reward, next_state, game_over))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            # exploration
            return np.random.randint(0, 2)
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
                target = reward + self.gamma * np.amax(self.policy_net(next_state))
            target_f = self.policy_net(state)
            target_f[0][action] = target
            self.policy_net.fit(state, target_f, epochs=1, verbose=0)

import keyboard
def test(game):
    agent = Agent()
    state = State.StateFrame(4)
    while True:
        episode_reward = 0
        _, reward, _ = game.frame_step(agent.act(state.get(game)))
        episode_reward += reward
        if keyboard.is_pressed("q"):
            break
