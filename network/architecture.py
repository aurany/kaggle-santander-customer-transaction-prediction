import torch
import torch.nn as nn
import torch.nn.functional as F


class LogisticModel(nn.Module):
    def __init__(self, input_size):
        super(LogisticModel, self).__init__()
        self.linear = torch.nn.Linear(input_size, 1)

    def forward(self, x):
        out = torch.sigmoid(self.linear(x)).view(-1)
        return out


class NetworkModel(nn.Module):
    def __init__(self, input_size):
        super(NetworkModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 300),
            nn.ReLU(),
            nn.Dropout(0.33),
            nn.Linear(300, 100),
            nn.ReLU(),
            nn.Dropout(0.33),
            nn.Linear(100, 1),
        )

    def forward(self,x):
        #output = torch.sigmoid(self.fc(x)).view(-1)
        output = self.fc(x).view(-1)
        return output
