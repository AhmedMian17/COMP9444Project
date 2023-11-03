import torch
import torch.nn as nn

class QNet(nn.module):
    def __init__(self):
        super(QNet, self).__init__()
        inputTensor = 1 * 7
        inputStates = 3
        hiddenNodes = 30

        self.inputToHiddenLayer = nn.Sequential(nn.Linear(inputTensor * inputStates, hiddenNodes), nn.Tanh())
        self.hiddenLayerToOutputLayer = nn.Sequential(nn.Linear(hiddenNodes, 1), nn.Tanh())
        
        
    def save():
        torch.save(QNet.state_dict(), 'Models/Saved Models/model.pth')

    def forward(self, input):
        output = self.inputToHiddenLayer(input)
        output = self.hiddenLayerToOutputLayer(output)

        return output
