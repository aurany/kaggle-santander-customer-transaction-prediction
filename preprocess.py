
import pandas as pd
import numpy as np
import torch

def get_data(csv):
    data = pd.read_csv(csv)
    return data

def prepare_features(data):
    features = data[['var_0', 'var_1']].values
    features = features.astype(np.float32)
    return features

def prepare_labels(data):
    labels = data['target'].values
    labels = labels.astype(np.float32)
    return labels

def prepare_data(csv):
    data = get_data(csv)

    features = prepare_features(data)
    labels = prepare_labels(data)

    # return TensorDataset(
    #     torch.from_numpy(features),
    #     torch.from_numpy(labels)
    # )

    return torch.from_numpy(features), torch.from_numpy(labels).long()

# X, y = prepare_data('data/train.csv')
# X.shape
# y.shape
