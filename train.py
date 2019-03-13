
import argparse

import numpy as np
import torch
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader, SubsetRandomSampler

from config import Config
from network import LogisticModel
from preprocess import get_data_loaders

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.contrib.metrics import ROC_AUC, AveragePrecision

import didactic_meme as model_suite

try:
    from tensorboardX import SummaryWriter
except ImportError:
    raise RuntimeError("No tensorboardX package is found. Please install with the command: \npip install tensorboardX")


def train(config):

    train_loader, val_loader, _ = get_data_loaders(config.dataset)

    writer = SummaryWriter(log_dir=f'{config.model_dir}/logs')

    n_features = train_loader.dataset[0][0].shape[0]
    model = LogisticModel(input_size=n_features)
    device = 'cpu'

    if torch.cuda.is_available():
        device = 'cuda'

    print(f'Device {device} will be used')

    loss = torch.nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=config.learning_rate)

    trainer = create_supervised_trainer(
        model,
        optimizer,
        loss,
        device=device
    )
    evaluator = create_supervised_evaluator(
        model,
        metrics={
            'loss': Loss(loss),
            'roc': ROC_AUC(),
            'accuracy': Accuracy(),
            'precision': AveragePrecision()
        },
        device=device
    )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        avg_loss = metrics['loss']
        avg_roc = metrics['roc']
        avg_accuracy = metrics['accuracy']
        avg_precision = metrics['precision']
        if engine.state.epoch % config.log_interval == 0:
            print("Training Results - Epoch: {}  Avg loss: {:.2f} ROC: {:.2f}, accuracy: {:.2f}, precision: {:.2f}"
                  .format(engine.state.epoch, avg_loss, avg_roc, avg_accuracy, avg_precision))
        writer.add_scalar("training/avg_loss", avg_loss, engine.state.epoch)
        writer.add_scalar("training/avg_roc", avg_roc, engine.state.epoch)
        writer.add_scalar("training/avg_accuracy", avg_accuracy, engine.state.epoch)
        writer.add_scalar("training/avg_precision", avg_precision, engine.state.epoch)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        avg_loss = metrics['loss']
        avg_roc = metrics['roc']
        avg_accuracy = metrics['accuracy']
        avg_precision = metrics['precision']
        if engine.state.epoch % config.log_interval == 0:
            print("Validation Results - Epoch: {}  Avg loss: {:.2f}, ROC: {:.2f}, accuracy: {:.2f}, precision: {:.2f}"
                  .format(engine.state.epoch, avg_loss, avg_roc, avg_accuracy, avg_precision))
        writer.add_scalar("valdation/avg_loss", avg_loss, engine.state.epoch)
        writer.add_scalar("valdation/avg_roc", avg_roc, engine.state.epoch)
        writer.add_scalar("valdation/avg_accuracy", avg_accuracy, engine.state.epoch)
        writer.add_scalar("valdation/avg_precision", avg_precision, engine.state.epoch)

    trainer.run(train_loader, max_epochs=config.n_epochs)
    writer.close()


if __name__ == '__main__':
	model_suite.command_line.train(Config, train)
