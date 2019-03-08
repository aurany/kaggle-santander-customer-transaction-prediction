
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset

def get_data(csv):
    data = pd.read_csv(csv, nrows=10000)
    return data

def prepare_features(data):
    features = data[[f'var_{i}' for i in range(10)]].values
    return features.astype(np.float32)

def prepare_labels(data):
    labels = data['target'].values
    return labels.astype(np.float32)

def prepare_data(csv):
    data = get_data(csv)

    features = prepare_features(data)
    labels = prepare_labels(data)

    return TensorDataset(
        torch.from_numpy(features).float(),
        torch.from_numpy(labels).float()
    )

# X, y = prepare_data('data/train.csv')
# X.shape
# y.shape
