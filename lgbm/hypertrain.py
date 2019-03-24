
import time
import logging
import argparse

import didactic_meme as model_suite

from lgbm.config import Config
from lgbm.train import get_data, train

logger = logging.getLogger(__name__)


def pretty_print_time_elapsed(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return '{:02}:{:02}:{:02}'.format(int(hours), int(minutes), int(seconds))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('runs_dir')
    args = parser.parse_args()

    data = get_data()

    hyperparameter_set = 1

    for num_leaves in [20, 30, 50]:
        for min_data_in_leaf in [60, 80, 120]:
            for learning_rate in [0.05, 0.1, 0.2]:
                for feature_fraction in [0.8, 0.9, 1.0]:
                    for bagging_fraction in [0.6, 0.8, 1.0]:
                        for lambda_l2 in [0, 0.01, 0.1]:

                            time_start = time.time()

                            config = Config()
                            config.num_leaves = num_leaves
                            config.min_data_in_leaf = min_data_in_leaf
                            config.learning_rate = learning_rate
                            config.feature_fraction = feature_fraction
                            config.bagging_fraction = bagging_fraction
                            config.lambda_l2 = lambda_l2

                            model_dir = f'{args.runs_dir}/hyper_{str(hyperparameter_set).zfill(4)}'
                            config.save(model_dir)
                            train(config, data=data)

                            time_elapsed = time.time() - time_start
                            time_elapsed_string = pretty_print_time_elapsed(time_elapsed)
                            logger.info(f'Time elapsed: {time_elapsed_string}')

                            hyperparameter_set += 1
