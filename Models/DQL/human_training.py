import Models.DQL.state as State
# import game.flappyNoGraphics as Game
import game.wrapped_flappy_bird as GameVisual
from collections import deque
import keyboard
import pickle

class Trainer(object):
    def __init__(self, agent):
        self.runs = 10
        self.agent = agent
        self.game = GameVisual.GameState()
        self.state_manager = State.StateManager(4)

    def play(self, runs=10):
        self.runs = runs
        self.agent.episodic_memory = []
        current_step = 0
        for i in range(self.runs):
            self.game = GameVisual.GameState()
            self.state_manager = State.StateManager(4)
            state = self.state_manager.get()
            done = False
            score = 0
            while not done:
                if keyboard.is_pressed(" "):
                    action = 1
                    _, reward, _ = self.game.frame_step(True)
                else:
                    action = 0
                    _, reward, _ = self.game.frame_step(False)
                if (reward == -1):
                    done = True
                    final_score = score
                    reward = -10
                score += reward

                self.state_manager.push(self.game)
                
                next_state = self.state_manager.get()
                self.agent.update_episodic_memory(state, action, reward, next_state, done, score, current_step)
                self.agent.learn()
                self.agent.learn_successful()
                current_step += 1
                state = next_state
            for frame in self.agent.episodic_memory:
                self.agent.remember(frame[0], frame[1], frame[2], frame[3], frame[4], frame[5], frame[6])
                self.agent.remember_successful(frame[0], frame[1], frame[2], frame[3], frame[4], frame[5], frame[6])
        




