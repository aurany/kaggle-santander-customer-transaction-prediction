
import logging

import pandas as pd
import numpy as np

from scipy.stats import binned_statistic

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler

from common.gaussrank import GaussRankScaler

from tqdm import tqdm

logger = logging.getLogger(__name__)


# def split_data(data_df):
#     data_df = data_df.copy()
#     train_df, val_df, test_df = np.split(
#         data_df.sample(frac=1),
#         [int(.6*len(data_df)), int(.8*len(data_df))]
#     )
#     return train_df, val_df, test_df


def read_data():

    logger.info(f'Reading data')

    train_df = pd.read_csv('data/train.csv')
    submission_df = pd.read_csv('data/test.csv')

    np.random.seed(seed=12378)
    random_numbers = np.random.uniform(0, 1, len(train_df))

    train_df['sample'] = 'train'
    train_df.loc[np.logical_and(random_numbers >= 0.6, random_numbers < 0.8), 'sample'] = 'validation'
    train_df.loc[(random_numbers >= 0.8), 'sample'] = 'test'
    submission_df['sample'] = 'submission'

    df = pd.concat((train_df, submission_df), axis=0, sort=False)

    return df


def get_onehot(x, y, train_idx):
    name = x.name
    x_train, y_train = x[train_idx], y[train_idx]

    x = x.values.reshape(-1, 1)
    x_train = x_train.values.reshape(-1, 1)
    y_train = y_train.values.reshape(-1, 1)

    model = DecisionTreeClassifier(max_depth=2, max_leaf_nodes=3, min_samples_leaf=1000)
    model.fit(x_train, y_train)
    predictions = model.predict_proba(x)[:, 1].reshape(-1, 1)

    encoder = OneHotEncoder(handle_unknown='ignore')
    encoder.fit(predictions)
    categories = encoder.transform(predictions).toarray()

    assert(np.isnan(predictions).sum() == 0)

    #logger.info(f'Preprocess {name} converted to {categories.shape[1]} group(s)')

    return categories


def prepare_features(data_df, train_idx):

    data_df = data_df.copy()

    original_features = [f'var_{i}' for i in range(200)]

    for feature in tqdm(original_features):

        lower = np.percentile(data_df[feature], 0.001)
        upper = np.percentile(data_df[feature], 99.999)
        data_df[feature] = data_df[feature].clip(lower, upper)

        data_df[f'{feature}_sq3'] = data_df[feature] ** 3
        data_df[f'{feature}_rt3'] = data_df[feature] ** 1/3

        #scaler = MinMaxScaler()
        #scaler = GaussRankScaler()
        scaler = StandardScaler()
        data_df[feature] = scaler.fit_transform(data_df[feature].values.reshape(-1, 1))
        data_df[f'{feature}_sq3'] = scaler.fit_transform(data_df[f'{feature}_sq3'].values.reshape(-1, 1))
        data_df[f'{feature}_rt3'] = scaler.fit_transform(data_df[f'{feature}_rt3'].values.reshape(-1, 1))

        var_bins = binned_statistic(data_df[feature], data_df[feature], bins=100, statistic='count')
        data_df[f'{feature}_grp'] = var_bins.binnumber

        var_to_onehot = get_onehot(data_df[feature], data_df['target'], train_idx)
        for column in range(var_to_onehot.shape[1]):
            data_df[f'{feature}_oh{column}'] = var_to_onehot[:, column]

    features = data_df.drop(['ID_code', 'target', 'sample'], axis=1)
    logger.info(f'Number of features {len(features.columns)}')

    return features.values.astype(np.float32)


def prepare_labels(data):
    labels = data['target'].values
    return labels.astype(np.float32)


def prepare_data(data_df):

    train_idx = (data_df['sample'] == 'train')
    val_idx = (data_df['sample'] == 'validation')
    test_idx = (data_df['sample'] == 'test')
    subm_idx = (data_df['sample'] == 'submission')

    X = prepare_features(data_df, train_idx)
    y = prepare_labels(data_df)

    return X[train_idx], y[train_idx], X[val_idx], y[val_idx], X[test_idx], y[test_idx], X[subm_idx]


#data_df = read_data()
#X_train, y_train, X_val, y_val, X_test, y_test, X_submission = prepare_data(data_df)
