
import numpy as np
import pandas as pd

import sklearn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from common.preprocess import read_data, prepare_data

X_train, y_train, X_val, y_val, X_test, y_test, X_submission = prepare_data(read_data())

n_components = 7

pca = PCA(n_components=n_components, random_state=12378)
pca.fit(X_train)
X_train_decomp = pca.transform(X_train)
X_val_decomp = pca.transform(X_val)

logistic = LogisticRegression(solver='lbfgs', C=0.01)

preds_train = []
preds_val = []

for component in range(n_components):
    x1 = X_train_decomp[:, component].reshape(-1, 1)
    x2 = (x1**2).reshape(-1, 1)
    x = np.concatenate([x1, x2], axis=1)
    logistic.fit(x, y_train)
    yhat_train = logistic.predict_proba(x)[:, 1].reshape(-1, 1)
    preds_train.append(yhat_train)

    x1 = X_val_decomp[:, component].reshape(-1, 1)
    x2 = (x1**2).reshape(-1, 1)
    x = np.concatenate([x1, x2], axis=1)
    yhat_val = logistic.predict_proba(x)[:, 1].reshape(-1, 1)
    preds_val.append(yhat_val)

X_train_preds = np.concatenate(preds_train, axis=1)
X_val_preds = np.concatenate(preds_val, axis=1)

df = pd.DataFrame(np.concatenate((X_train_preds, y_train.reshape(-1, 1)), axis=1))
df.columns = [f'var{i}' for i in range(7)]+['target']
df.head()

from random import shuffle

vars = ['var0', 'var1', 'var2', 'var3', 'var4', 'var5', 'var6']

for _ in range(100):

    shuffle(vars)
    df_sorted = df.sort_values(by=vars, ascending=True, axis=0)
    df_sorted.head()

    df_sorted = df_sorted.reset_index()
    df_sorted.head()

    df_sorted['y_pred'] = df_sorted.index
    df_sorted

    auc = roc_auc_score(df_sorted['target'], df_sorted['y_pred'])
    print(auc)
