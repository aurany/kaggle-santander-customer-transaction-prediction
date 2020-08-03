
import logging

import numpy as np
import lightgbm as lgb

from common.preprocess import read_data, prepare_data
from lgbm.config import Config

import didactic_meme as model_suite

try:
    from tensorboardX import SummaryWriter
except ImportError:
    raise RuntimeError("No tensorboardX package is found. Please install with the command: \npip install tensorboardX")

logger = logging.getLogger(__name__)


def tb_logger(writer):
    def callback(env):
        for ds, metric, value, _ in env.evaluation_result_list:
            if env.iteration % 100 == 0:
                writer.add_scalar(f'{ds}/{metric}', value, env.iteration)
    return callback

def get_data():
    data_df = read_data()
    return prepare_data(data_df)

def train(config, data=None):

    model_suite.logging.setup_loggers(config)
    writer = SummaryWriter(log_dir=f'{config.model_dir}/logs')

    if data == None:
        X_train, y_train, X_val, y_val, X_test, y_test, X_submission = get_data()
    else:
        X_train, y_train, X_val, y_val, X_test, y_test, X_submission = [d for d in data]

    logger.info(f'Data imported')

    train_ds = lgb.Dataset(X_train, label=y_train)
    val_ds = lgb.Dataset(X_val, label=y_val)
    logger.info(f'Datasets created')

    params = config.to_dict()
    logger.info(params)

    gbm = lgb.train(
        params,
        train_ds,
        150000,
        valid_sets=[train_ds, val_ds],
        callbacks=[tb_logger(writer)],
        verbose_eval=100,
        early_stopping_rounds=5000
    )
    logger.info(f'Training finished')

    gbm.save_model(f'{config.model_dir}/model.txt')

    writer.close()

if __name__ == '__main__':
	model_suite.command_line.train(Config, train)
