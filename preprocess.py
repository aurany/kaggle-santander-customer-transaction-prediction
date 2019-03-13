
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import RandomSampler

def get_data(csv):
    data = pd.read_csv(csv)
    return data

def prepare_features(data):
    features = data[[f'var_{i}' for i in range(10)]].values
    return features.astype(np.float32)

def prepare_labels(data):
    labels = data['target'].values
    return labels.astype(np.float32)

def split_data(data_df):
    data_df = data_df.copy()
    train_df, val_df, test_df = np.split(data_df.sample(frac=1), [int(.6*len(data_df)), int(.8*len(data_df))])
    return train_df, val_df, test_df

def get_data_loaders(csv, train_batch_size=1000, val_batch_size=2000):
    data_df = get_data(csv)
    train_df, val_df, test_df = split_data(data_df)
    train_ds, val_ds, test_ds = [
        TensorDataset(
            torch.from_numpy(prepare_features(df)).float(),
            torch.from_numpy(prepare_labels(df)).float()
        )
        for df in [train_df, val_df, test_df]
    ]

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
    test_loader = torch.utils.data.DataLoader(
        test_ds
    )

    return train_loader, val_loader, test_loader

# train_loader, val_loader, test_loader = get_data_loaders('data/train.csv')
