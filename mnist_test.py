from time import time

import numpy as np
import pandas as pd
import sklearn.metrics as sk_metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10

from nam.wrapper import NAMClassifier, MultiTaskNAMClassifier


def make_gender_mtl_data(X, y):
    y_male = y.copy()
    y_male[X['sex'] == 1] = np.nan
    y_female = y.copy()
    y_female[X['sex'] == 0] = np.nan
    return pd.concat([y_female, y_male], axis=1)

def filterClasses(X, y, classes:list):
    X = X[np.isin(y[:, 0], classes)]
    y = y[np.isin(y[:, 0], classes)]
    return X, y

def cood_encoding(x):
    x = x / 255
    indices = np.zeros((x.shape[1], x.shape[2], 2))
    for i in range(x.shape[1]):
        for j in range(x.shape[2]):
            indices[i, j] = [i, j]
    indices = np.repeat(indices[np.newaxis, :, :, :], x.shape[0], axis=0)
    print(x.shape, indices.shape)
    return np.concatenate((x, indices), axis=3).reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3] + 2)
    # return x.reshape(x.shape[0], x.shape[1] * x.shape[2], 1)

if __name__ == '__main__':
    random_state = 2016
    dataset = CIFAR10(root='nam/data/', download=True, train=True)
    # X_data = dataset.data.numpy()
    # y_data = dataset.targets.numpy().reshape(-1, 1)
    X_data = dataset.data
    y_data = np.array(dataset.targets).reshape(-1, 1)

    X_data, y_data = filterClasses(X_data, y_data, [0, 1])
    # y_data = np.max(1, y_data, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, train_size=0.2, test_size=0.8, random_state=random_state, stratify=y_data)
    X_test, _, y_test, _ = train_test_split(X_test, y_test, train_size=0.1, test_size=0.9, random_state=random_state, stratify=y_test)

    X_train = cood_encoding(X_train)
    X_test = cood_encoding(X_test)
    print(X_train.shape, X_test.shape)

    s_time = time()
    model = NAMClassifier(
        num_epochs=40,
        num_learners=1,
        lr=0.01,
        batch_size=256,
        metric='auroc',
        early_stop_mode='max',
        monitor_loss=False,
        n_jobs=10,
        random_state=random_state,
        device='cuda:0'
    )
    
    model.fit(X_train, y_train)

    pred = model.predict_proba(X_test)
    
    print(pred)
    print("----------")
    print(y_test)

    e_time = time()
    print(sk_metrics.roc_auc_score(y_test, pred))
    print("Time cost: " + str(e_time - s_time))


