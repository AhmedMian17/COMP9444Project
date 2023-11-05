import torch
import torch.nn as nn

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__(self)
        model = nn.Sequential({
            nn.Linear(7*4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Sigmoid(),
        })
        # TODO
        return 
    def forward(self, x):
        # TODO
        output = self.model(x)
        return output