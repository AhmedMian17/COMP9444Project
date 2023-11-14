import torch
import torch.nn as nn
import torch.optim as optim

class Network(nn.Module):
    def __init__(self, lr):
        super(Network, self).__init__()

        self.n_actions = 2
        self.hid_1 = 128
        self.hid_2 = 128
        self.hid_3 = 128
        self.inputs = 7 * 4
        self.model = nn.Sequential(
            nn.Linear(self.inputs, self.hid_1),
            nn.ReLU(),
            nn.Linear(self.hid_1, self.hid_2),
            nn.ReLU(),
            nn.Linear(self.hid_2, self.hid_3),
            nn.ReLU(),
            nn.Linear(self.hid_3, self.n_actions),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.SmoothL1Loss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        output = self.model(x.type(torch.FloatTensor).to(self.device))
        return output