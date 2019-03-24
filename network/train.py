
import logging
import argparse

import numpy as np
import torch
from torch.optim import SGD, Adam
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import RandomSampler

from network.config import Config
from network.architecture import LogisticModel, NetworkModel
from common.preprocess import read_data, prepare_data

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.contrib.metrics import ROC_AUC, AveragePrecision

import didactic_meme as model_suite

try:
    from tensorboardX import SummaryWriter
except ImportError:
    raise RuntimeError("No tensorboardX package is found. Please install with the command: \npip install tensorboardX")

logger = logging.getLogger(__name__)


def get_model(model_name, n_features):
    if model_name == 'Logistic':
        model = LogisticModel(input_size=n_features)
    elif model_name == 'Network':
        model = NetworkModel(input_size=n_features)
    else:
        raise NotImplementedError()
    return model

def get_datasets(data_df):
    X_train, y_train, X_val, y_val, X_test, y_test, X_submission = prepare_data(data_df)

    train_ds = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).float()
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val).float(),
        torch.from_numpy(y_val).float()
    )
    test_ds = TensorDataset(
        torch.from_numpy(X_val).float(),
        torch.from_numpy(y_val).float()
    )
    return train_ds, val_ds, test_ds

def get_data_loaders(train_ds, val_ds, train_batch_size=1000, val_batch_size=2000):
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        sampler=RandomSampler(train_ds),
        batch_size=train_batch_size
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        sampler=RandomSampler(val_ds),
        batch_size=val_batch_size
    )

    return train_loader, val_loader

def train(config):

    model_suite.logging.setup_loggers(config)

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    logger.info(f'Device {device} will be used')

    data_df = read_data()
    train_ds, val_ds, test_ds = get_datasets(data_df)
    train_loader, val_loader = get_data_loaders(
        train_ds,
        val_ds,
        train_batch_size=config.train_batch_size,
        val_batch_size=config.val_batch_size
    )

    writer = SummaryWriter(log_dir=f'{config.model_dir}/logs')

    n_features = train_loader.dataset[0][0].shape[0]
    model = get_model(model_name=config.model, n_features=n_features)
    loss = torch.nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    #optimizer = SGD(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, momentum=config.momentum)

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
        # avg_accuracy = metrics['accuracy']
        # avg_precision = metrics['precision']
        logger.info(f'Training results - Epoch: {engine.state.epoch} Avg loss: {avg_loss} ROC: {avg_roc}')
        writer.add_scalar("training/avg_loss", avg_loss, engine.state.epoch)
        writer.add_scalar("training/avg_roc", avg_roc, engine.state.epoch)
        # writer.add_scalar("training/avg_accuracy", avg_accuracy, engine.state.epoch)
        # writer.add_scalar("training/avg_precision", avg_precision, engine.state.epoch)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        avg_loss = metrics['loss']
        avg_roc = metrics['roc']
        # avg_accuracy = metrics['accuracy']
        # avg_precision = metrics['precision']
        logger.info(f'Validation results - Epoch: {engine.state.epoch} Avg loss: {avg_loss} ROC: {avg_roc}')
        writer.add_scalar("valdation/avg_loss", avg_loss, engine.state.epoch)
        writer.add_scalar("valdation/avg_roc", avg_roc, engine.state.epoch)
        # writer.add_scalar("valdation/avg_accuracy", avg_accuracy, engine.state.epoch)
        # writer.add_scalar("valdation/avg_precision", avg_precision, engine.state.epoch)

    trainer.run(train_loader, max_epochs=config.n_epochs)
    writer.close()


if __name__ == '__main__':
	model_suite.command_line.train(Config, train)
