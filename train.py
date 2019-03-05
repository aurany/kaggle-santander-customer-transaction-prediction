
from preprocess import prepare_data
from network import LogisticRegression
import torch
from torch import nn
from skorch.classifier import NeuralNet


net = NeuralNet(
    LogisticRegression(input_size=2),
    max_epochs=10,
    lr=0.01,
    iterator_train__shuffle=True,
    criterion=nn.NLLLoss
)


X, y = prepare_data('data/train.csv')
net.fit(X, y)


#y_proba = net.predict_proba(X)
