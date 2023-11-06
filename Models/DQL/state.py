from collections import deque
import torch
from utils import get_input_layer as input

class StateFrame(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        starting_state = [0, 0, 0, 0, 0, 0, 0]
        for _ in range(capacity):
            self.memory.append(starting_state)

    def push(self, game):
        """Save a frame, 
            returns tensor of flattened state frames
        """
        state_frame = input(game)
        self.memory.popleft()
        self.memory.append(state_frame)
        tensor_list = []
        for i in range(4):
            tensor_list.append(torch.tensor(self.memory[i]))
        return torch.stack(tensor_list, dim=0)
    
    def get(self):
        tensor_list = []
        for i in range(4):
            tensor_list.append(torch.tensor(self.memory[i]))
        return torch.stack(tensor_list, dim=0)