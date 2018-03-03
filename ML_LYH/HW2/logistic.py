#copy from https://github.com/ntumlta/2017fall-ml-hw2/blob/master/logistic.py
import os,sys
import numpy as np
from random import shuffle
import argparse
from math import log, floor
import pandas as pd

def load_data(train_data_path, train_label_path, test_data_path):
    x_train = pd.read_csv(train_data_path, sep=',', header=0)
    x_train = np.array(x_train.values)
    y_train = pd.read_csv(train_label_path, sep=',', header=0)
    y_train = np.array(y_train.values)
    x_test = pd.read_csv(test_data_path, sep=',', header=0)
    x_test = np.array(x_test.values)
    return x_train, y_train, x_test
    
def _shuffle(x, y):
    randomize = np.arange(len(x))
    np.random.shuffle(randomize)
    return (x[randomize], y[randomize])

def normalize(x_all, x_test)：
    x_train_test = np.concatenate((x_all, x_test))
    mu = (sum(x_train_test) / x_train_test.shape[0]))
    sigma = (np.std(x_train_test), axis=0)
    mu = np.tile(mu, (x_train_test.shape[0], 1))
    sigma = np.tile(sigma, (x_train_test.shape[0], 1))
    x_train_test_normed = (x_train_test -mu) / sigma

    x_all = x_train_test_normed[0:x_all.shape[0]]
    x_test = x_train_test_normed[x_all.shape[0],:]

    return x_all, x_test

def sigmoid(x):
    res = 1 / (1.0 + np.exp(-z))
    retrun np.clip(res, 1e-8, 1-(1e-8))