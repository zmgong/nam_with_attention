from time import time

import numpy as np
import pandas as pd
import sklearn.metrics as sk_metrics
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import random_split

from nam.wrapper import NAMClassifier, MultiTaskNAMClassifier


def make_gender_mtl_data(X, y):
    y_male = y.copy()
    y_male[X['sex'] == 1] = np.nan
    y_female = y.copy()
    y_female[X['sex'] == 0] = np.nan
    return pd.concat([y_female, y_male], axis=1)

def onehot_pos_embedding(x):
    onehot = np.zeros((6, 6))
    for i in range(6):
        onehot[i, i] = 1
    onehot = np.repeat(onehot[np.newaxis, :, :], x.shape[0], axis=0)
    return np.concatenate((x[:, :, np.newaxis], onehot), axis=2)

if __name__ == '__main__':

    pre_train_path = None

    random_state = 2016
    dataset = pd.read_csv('nam/data/recid.data', delimiter=' ', header=None)
    dataset.columns = ["age", "race", "sex", "priors_count", "length_of_stay", "c_charge_degree", "two_year_recid"]
    binary = ['sex', 'c_charge_degree']
    other = ['age', 'race', 'priors_count', 'length_of_stay']
    scaler = MinMaxScaler((-1, 1))
    dataset[other] = scaler.fit_transform(dataset[other])
    dataset[binary] = dataset[binary] - 1

    X_data = dataset.drop(columns=['two_year_recid']).to_numpy()
    y_data = dataset['two_year_recid'].to_numpy().reshape(-1, 1)
    X_data = onehot_pos_embedding(X_data)
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, train_size=0.8, test_size=0.2, random_state=random_state)

    #Single Task NAMs Classification
    s_time = time()
    model = NAMClassifier(
        num_epochs=1000,
        num_learners=20,
        metric='auroc',
        early_stop_mode='max',
        monitor_loss=False,
        n_jobs=10,
        random_state=random_state,
        device='cuda:0'
    )
    
    model.fit(X_train, y_train)


    pred, weight, att_weight = model.predict_proba(X_test)

    e_time = time()
    print(sk_metrics.roc_auc_score(y_test, pred))
    print("Time cost: " + str(e_time - s_time))





