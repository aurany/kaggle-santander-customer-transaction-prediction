
import argparse

import numpy as np
import torch
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader, SubsetRandomSampler

from network import LogisticModel
from preprocess import prepare_data

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.contrib.metrics import ROC_AUC

from sampler import ImbalancedDatasetSampler

try:
    from tensorboardX import SummaryWriter
except ImportError:
    raise RuntimeError("No tensorboardX package is found. Please install with the command: \npip install tensorboardX")


def create_summary_writer(model, data_loader, log_dir):
    writer = SummaryWriter(log_dir=log_dir)
    data_loader_iter = iter(data_loader)
    X, y = next(data_loader_iter)
    try:
        writer.add_graph(model, X)
    except Exception as e:
        print("Failed to save model graph: {}".format(e))
    return writer

def run(train_ds, val_ds, batch_size, epochs, lr, weight_decay, log_interval, log_dir):

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        sampler=ImbalancedDatasetSampler(train_ds),
        batch_size=batch_size
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    n_features = train_ds[0][0].shape[0]
    model = LogisticModel(input_size=n_features)
    writer = create_summary_writer(model, train_loader, log_dir)
    device = 'cpu'

    if torch.cuda.is_available():
        device = 'cuda'

    loss = torch.nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=lr)
    trainer = create_supervised_trainer(
        model,
        optimizer,
        loss,
        device=device
    )
    evaluator = create_supervised_evaluator(
        model,
        metrics={
            'accuracy': Accuracy(),
            'bce': Loss(loss),
            'roc': ROC_AUC()
        },
        device=device
    )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_bce = metrics['bce']
        avg_roc = metrics['roc']
        if engine.state.epoch % log_interval == 0:
            print("Training Results - Epoch: {}  Avg accuracy: {:.2f}, loss: {:.2f} ROC: {:.2f}"
                  .format(engine.state.epoch, avg_accuracy, avg_bce, avg_roc))
        writer.add_scalar("training/avg_loss", avg_bce, engine.state.epoch)
        writer.add_scalar("training/avg_accuracy", avg_accuracy, engine.state.epoch)
        writer.add_scalar("training/avg_roc", avg_roc, engine.state.epoch)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_bce = metrics['bce']
        avg_roc = metrics['roc']
        if engine.state.epoch % log_interval == 0:
            print("Validation Results - Epoch: {}  Avg accuracy: {:.2f}, loss: {:.2f}, ROC: {:.2f}"
                  .format(engine.state.epoch, avg_accuracy, avg_bce, avg_roc))
        writer.add_scalar("valdation/avg_loss", avg_bce, engine.state.epoch)
        writer.add_scalar("valdation/avg_accuracy", avg_accuracy, engine.state.epoch)
        writer.add_scalar("valdation/avg_roc", avg_roc, engine.state.epoch)

    trainer.run(train_loader, max_epochs=epochs)
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=2000,
                        help='input batch size for training (default: 2000)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='how many batches to wait before logging training status')
    parser.add_argument("--log_dir", type=str, default="tensorboard_logs",
                        help="log directory for Tensorboard log output")

    args = parser.parse_args()
    train_ds, val_ds, test_ds = prepare_data('data/train.csv')

    run(
        train_ds,
        val_ds,
        args.batch_size,
        args.epochs,
        args.lr,
        args.weight_decay,
        args.log_interval,
        args.log_dir
    )
