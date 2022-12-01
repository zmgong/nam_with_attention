from time import time

import numpy as np
import pandas as pd
import sklearn.metrics as sk_metrics
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
from tqdm import tqdm

from nam.wrapper import NAMClassifier, MultiTaskNAMClassifier

from numpy import genfromtxt
import nltk

from torch.nn.functional import pad

from nltk.tokenize import word_tokenize, sent_tokenize
import gensim
from gensim.models import Word2Vec


def filterClasses(X, y, classes: list):
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


def tokenize_strs(str_array):
    token_list = []
    max_len = 0
    for i in tqdm(str_array):
        curr_token = word_tokenize(i)
        token_list.append(curr_token)
        curr_len = len(curr_token)
        if curr_len > max_len:
            max_len = curr_len
    print(max_len)
    for index, i in enumerate(tqdm(token_list)):
        num_of_pad = max_len - len(i)
        pads = np.array(['[PAD]'] * num_of_pad)
        tokens = np.array(i)
        tokens = np.concatenate([tokens, pads])
        token_list[index] = tokens
    token_list = np.array(token_list)
    print(token_list.shape)
    data = token_list.flatten()
    print("?")
    model1 = gensim.models.Word2Vec(token_list, min_count=1,
                                    vector_size=100, window=5)
    print(model1.wv['review'])
    exit()
    pass



if __name__ == '__main__':
    random_state = 2016
    # IMDB dataset download from https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews?resource=download
    dataset = pd.read_csv('nam/data/IMDB Dataset.csv', delimiter=',')

    review = dataset['review'].to_numpy()
    X_data = tokenize_strs(review)
    y_data = dataset['sentiment'].to_numpy()
    print(y_data.shape)
    y_data = y_data.reshape(-1, 1)
    print(y_data.shape)

    # Convert to embeddings

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, train_size=0.2, test_size=0.8,
                                                        random_state=random_state, stratify=y_data)
    # X_test, _, y_test, _ = train_test_split(X_test, y_test, train_size=0.1, test_size=0.9, random_state=random_state, stratify=y_test)
    #
    # X_train = cood_encoding(X_train)
    # X_test = cood_encoding(X_test)
    # print(X_train.shape, X_test.shape)
    #
    # s_time = time()
    # model = NAMClassifier(
    #     num_epochs=40,
    #     num_learners=1,
    #     lr=0.01,
    #     batch_size=64,
    #     metric='auroc',
    #     early_stop_mode='max',
    #     monitor_loss=False,
    #     n_jobs=10,
    #     random_state=random_state,
    #     device='cuda:0'
    # )
    #
    # model.fit(X_train, y_train)
    #
    # pred = model.predict_proba(X_test)
    #
    # print(pred)
    # print("----------")
    # print(y_test)
    #
    # e_time = time()
    # print(sk_metrics.roc_auc_score(y_test, pred))
    # print("Time cost: " + str(e_time - s_time))
