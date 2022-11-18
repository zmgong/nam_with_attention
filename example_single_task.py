from time import time

import numpy as np
import pandas as pd
import sklearn.metrics as sk_metrics
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

if __name__ == '__main__':
    random_state = 2016
    dataset = pd.read_csv('nam/data/recid.data', delimiter=' ', header=None)
    dataset.columns = ["age", "race", "sex", "priors_count", "length_of_stay", "c_charge_degree", "two_year_recid"]
    binary = ['sex', 'c_charge_degree']
    other = ['age', 'race', 'priors_count', 'length_of_stay']
    scaler = MinMaxScaler((-1, 1))
    dataset[other] = scaler.fit_transform(dataset[other])
    dataset[binary] = dataset[binary] - 1
    data_train, data_test = train_test_split(dataset, train_size=0.8, test_size=0.2, random_state=random_state)
    X_train, y_train = data_train[other + binary], data_train['two_year_recid']
    X_test, y_test = data_test[other + binary], data_test['two_year_recid']
    # Single Task NAMs Classification
    # s_time = time()
    # model = NAMClassifier(
    #     num_epochs=1000,
    #     num_learners=20,
    #     metric='auroc',
    #     early_stop_mode='max',
    #     monitor_loss=False,
    #     n_jobs=10,
    #     random_state=random_state
    # )
    #
    # model.fit(X_train, y_train)
    #
    # pred = model.predict_proba(X_test)
    # e_time = time()
    # print(sk_metrics.roc_auc_score(y_test, pred))
    # print("Time cost: " + str(e_time - s_time))

    # Multitask NAMs Classification
    y_train_mtl = make_gender_mtl_data(X_train, y_train)
    y_test_mtl = make_gender_mtl_data(X_test, y_test)

    X_train_mtl = X_train.drop(columns=['sex'])
    X_test_mtl = X_test.drop(columns=['sex'])
    s_time = time()
    model = MultiTaskNAMClassifier(
        num_learners=20,
        patience=60,
        num_epochs=1000,
        num_subnets=10,
        metric='auroc',
        monitor_loss=False,
        early_stop_mode='max',
        n_jobs=10,
        random_state=random_state
    )
    model.fit(X_train_mtl, y_train_mtl)
    pred = model.predict_proba(X_test_mtl)
    e_time = time()
    # Flatten and remove nans
    y_test_mtl_flat = y_test_mtl.to_numpy().reshape(-1)
    pred_flat = pred.reshape(-1)

    non_nan_indices = y_test_mtl_flat == y_test_mtl_flat
    y_test_mtl_flat = y_test_mtl_flat[non_nan_indices]
    pred_flat = pred_flat[non_nan_indices]
    print(sk_metrics.roc_auc_score(y_test_mtl_flat, pred_flat))
    print("Time cost: " + str(e_time - s_time))




