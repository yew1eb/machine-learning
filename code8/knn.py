#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from math import sqrt


def euclid_dist(d1, d2, ndim):
    return sqrt(np.sum([(d1[n] - d2[n]) ** 2 for n in range(ndim)]))


def classify(k_nn, ltrain):
    ltest = np.ravel(ltrain[k_nn]).astype(np.int64)
    return np.argmax(np.bincount(ltest))


def knn_predict(dtrain, dtest, ltrain, k=3):
    ndata, ndim = dtrain.shape

    ltest = []
    for dtst in dtest:
        # calculate distances between train and test
        dists = [euclid_dist(dtrn, dtst, ndim) for dtrn in dtrain]

        # choose k neareest neighbors
        k_nn = np.argsort(dists)[:k]

        # classify and save labels
        ltest.append(classify(k_nn, ltrain))

    return np.array(ltest)


def main():
    # setup sample data
    lnum = 2
    colors = ['r', 'b']
    dtrain = np.random.randint(0, 100, (50, 2)).astype(np.float32)
    ltrain = np.random.randint(0, lnum, (50, 1)).astype(np.float32)
    dtest = np.random.randint(0, 100, (5, 2)).astype(np.float32)

    # plot train data
    red = dtrain[ltrain.ravel() == 0]
    blue = dtrain[ltrain.ravel() == 1]
    plt.scatter(red[:, 0], red[:, 1], c='r')
    plt.scatter(blue[:, 0], blue[:, 1], c='b', marker='x')

    # predcit test data into class
    ltest = knn_predict(dtrain, dtest, ltrain, 3)
    for i in range(lnum):
        plt.scatter(dtest[ltest == i, 0],
                    dtest[ltest == i, 1],
                    color=colors[i], marker='D')

    # show results
    plt.show()


if __name__ == '__main__':
    main()
