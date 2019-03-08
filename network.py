
import torch

class LogisticModel(torch.nn.Module):
    def __init__(self, input_size):
        super(LogisticModel, self).__init__()
        self.linear = torch.nn.Linear(input_size, 1)

    def forward(self, x):
        out = torch.sigmoid(self.linear(x)).view(-1)
        return out
