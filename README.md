
# TODO


[ ] Hypertrain
[ ] Finalize features
[ ] More architectures




0.85
{
    "n_epochs": 500,
    "train_batch_size": 2000,
    "val_batch_size": 2000,
    "learning_rate": 0.005,
    "weight_decay": 0.005,
    "momentum": 0.001,
    "model": "Network",
    "seed": 7240605057070392132
}

class NetworkModel(nn.Module):
    def __init__(self, input_size):
        super(NetworkModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 400),
            nn.ReLU(inplace=True),
            nn.Dropout(0.20),
            nn.Linear(400, 200),
            nn.ReLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(200, 50),
            nn.ReLU(inplace=True),
            nn.Linear(50, 1),
        )

    def forward(self,x):
        #output = torch.sigmoid(self.fc(x)).view(-1)
        output = self.fc(x).view(-1)
        return output
