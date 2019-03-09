
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset

def get_data(csv):
    data = pd.read_csv(csv)
    return data

def prepare_features(data):
    features = data[[f'var_{i}' for i in range(10)]].values
    return features.astype(np.float32)

def prepare_labels(data):
    labels = data['target'].values
    return labels.astype(np.float32)

def prepare_data(csv):
    data_df = get_data(csv)

    train_df, val_df, test_df = np.split(data_df.sample(frac=1), [int(.6*len(data_df)), int(.8*len(data_df))])

    return [
        TensorDataset(
            torch.from_numpy(prepare_features(df)).float(),
            torch.from_numpy(prepare_labels(df)).float()
        )
        for df in [train_df, val_df, test_df]
    ]

#train_ds, val_ds, test_ds = prepare_data('data/train.csv')
