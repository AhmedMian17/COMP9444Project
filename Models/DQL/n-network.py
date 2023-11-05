import torch
import torch.nn as nn

class Network(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        model = nn.Sequential({
            nn.Linear(7*4, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2),
            nn.Sigmoid(),
        })
        # TODO
        return 
    def forward(self, x):
        # TODO
        output = self.model(x)
        return output