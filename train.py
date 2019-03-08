
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

try:
    from tensorboardX import SummaryWriter
except ImportError:
    raise RuntimeError("No tensorboardX package is found. Please install with the command: \npip install tensorboardX")


def run(data, batch_size, epochs, lr, weight_decay, log_interval, log_dir):

    random_seed = 123457
    validation_split = 0.2
    n_obs = len(data)
    indices = list(range(n_obs))
    split = int(np.floor(validation_split * n_obs))
    np.random.seed(random_seed)
    np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(data, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(data, batch_size=batch_size, sampler=val_sampler)

    n_features = data[0][0].shape[0]
    model = LogisticModel(input_size=n_features)
    #writer = create_summary_writer(model, train_loader, log_dir)
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
        # writer.add_scalar("training/avg_loss", avg_bce, engine.state.epoch)
        # writer.add_scalar("training/avg_accuracy", avg_accuracy, engine.state.epoch)
        # writer.add_scalar("training/avg_roc", avg_roc, engine.state.epoch)

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
        # writer.add_scalar("valdation/avg_loss", avg_bce, engine.state.epoch)
        # writer.add_scalar("valdation/avg_accuracy", avg_accuracy, engine.state.epoch)
        # writer.add_scalar("valdation/avg_roc", avg_roc, engine.state.epoch)

    trainer.run(train_loader, max_epochs=epochs)
    #writer.close()

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
    prepared_data = prepare_data('data/train.csv')

    run(
        prepared_data,
        args.batch_size,
        args.epochs,
        args.lr,
        args.weight_decay,
        args.log_interval,
        args.log_dir
    )
