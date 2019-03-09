
%load_ext autoreload
%autoreload

from preprocess import prepare_data
import seaborn as sns

# number of trues/nobs
train_ds, val_ds, test_ds = prepare_data('data/train.csv')

for ds in [train_ds, val_ds, test_ds]:
    X = ds[:][0].numpy()
    y = ds[:][1].numpy()

    print(f'number of obs {y.shape[0]}')
    print(f'number of true {(y==1).sum()}')
    print(f'percentage true {(y==1).sum()/y.shape[0]}')

# TBC...
