
%load_ext autoreload
%autoreload 2

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
from scipy.stats import binned_statistic

from common.preprocess import read_data, prepare_data

# number of trues/nobs
data_df = read_data()
data_df = data_df[data_df['sample'] == 'train']
X = data_df[[f'var_{i}' for i in range(200)]].values
y = data_df['target'].values

print(f'number of obs {y.shape[0]}')
print(f'number of true {(y==1).sum()}')
print(f'percentage true {(y==1).sum()/y.shape[0]}')

n_bins = 10
for var_index in range(10):

    _X = X[:, var_index]
    _X = (_X - _X.mean())/_X.std()
    _p5 = np.percentile(_X, 1)
    _p95 = np.percentile(_X, 99)
    _X = np.clip(_X, _p5, _p95)


    bin_means = binned_statistic(_X, y, bins=n_bins, statistic='mean')
    bin_count = binned_statistic(_X, y, bins=n_bins, statistic='count')

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.bar(range(n_bins), bin_count.statistic)
    plt.twinx()
    plt.plot(range(n_bins), bin_means.statistic)
    ax.set_title(f'var_{var_index}')
    plt.show()
# TBC...
