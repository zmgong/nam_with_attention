from time import time

import numpy as np
import pandas as pd
import sklearn.metrics as sk_metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10

from nam.wrapper import NAMClassifier, MultiTaskNAMClassifier
from CIFAR10_test import cood_encoding
import random


def make_gender_mtl_data(X, y):
    y_male = y.copy()
    y_male[X['sex'] == 1] = np.nan
    y_female = y.copy()
    y_female[X['sex'] == 0] = np.nan
    return pd.concat([y_female, y_male], axis=1)


def filterClasses(X, y, classes: list):
    X = X[np.isin(y[:, 0], classes)]
    y = y[np.isin(y[:, 0], classes)]
    return X, y


def single_training(X_train, X_test, y_train, y_test, cfg, epoch=40, random_seed=42):
    s_time = time()
    model = NAMClassifier(
        num_epochs=epoch,
        num_learners=1,
        lr=cfg["learning_rate"],
        batch_size=cfg["batchs_size"],
        metric='auroc',
        early_stop_mode='max',
        monitor_loss=False,
        n_jobs=10,
        random_state=random_seed,
        device='cuda:0',
        dropout=cfg["drop_out"],
        feature_dropout=cfg["feature_dropout"],
        pos_embed=3,
    )
    print(X_train.shape)
    print(y_train.shape)
    model.fit(X_train, y_train)

    pred, _, _, _ = model.predict_proba(X_test)

    # print(pred)
    print("----------")
    # print(y_test)

    e_time = time()
    score = sk_metrics.roc_auc_score(y_test, pred)
    print(score)
    print("Time cost: " + str(e_time - s_time))

    return score


if __name__ == '__main__':
    random_state = 42
    dataset = CIFAR10(root='nam/data/', download=True, train=True)
    # X_data = dataset.data.numpy()
    # y_data = dataset.targets.numpy().reshape(-1, 1)
    X_data = dataset.data
    y_data = np.array(dataset.targets).reshape(-1, 1)

    X_data, y_data = filterClasses(X_data, y_data, [0, 1])
    # y_data = np.max(1, y_data, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, train_size=0.2, test_size=0.8,
                                                        random_state=random_state, stratify=y_data)
    X_test, _, y_test, _ = train_test_split(X_test, y_test, train_size=0.1, test_size=0.9, random_state=random_state,
                                            stratify=y_test)

    X_train = cood_encoding(X_train)
    X_test = cood_encoding(X_test)
    print(X_train.shape, X_test.shape)
    best_score = 0
    best_cfg = None
    for seed in range(30):
        print("Current seed: " + str(seed))
        random.seed(seed)
        random_config = {
            "learning_rate": random.uniform(0.01, 0.0005),
            "drop_out": random.uniform(0.1, 0.001),
            "feature_dropout": random.uniform(0.1, 0.001),
            "batchs_size": random.choice([64, 128, 256])
        }
        score = single_training(X_train, X_test, y_train, y_test, random_config, epoch=40)
        if score > best_score:
            best_score = score
            best_cfg = random_config
            print("New best score is: " + str(best_score))
            print(best_cfg)
            print()
    print("Best score is: " + str(best_score))
    print(best_cfg)
    score = single_training(X_train, X_test, y_train, y_test, best_cfg, epoch=320)
    print("Final score is: " + str(score))
    print(best_cfg)
