#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
import multivariate_normal


def calculate_center(data_i):
    val = 0.0
    cnt_i = len(data_i)
    for d in data_i:
        val += d
    return (val / cnt_i)


def euclid_dist(center, data_i):
    return sqrt(sum([
        (center[n] - data_i[n]) ** 2 for n in range(len(center))
    ]))


def assign_label(center, data_i):
    dist = []
    for k in range(len(center)):
        dist.append(euclid_dist(center[k], data_i))
    return np.argsort(np.ravel(dist))[0]


def kmeans(data, k, max_iter=300):
    # assign cluster label to each data at random
    labels = np.random.randint(0, k, len(data))

    iter = 0
    center = np.zeros([k, 2])
    while 1:
        # calculate the cluster centers
        for i in range(k):
            center[i] = calculate_center(data[labels == i])

        # assign new label
        new_labels = np.zeros(len(data)).astype(np.int)
        for i, d in enumerate(data):
            new_labels[i] = assign_label(center, d)

        if np.array_equal(labels, new_labels) or iter > max_iter:
            break
        else:
            labels = new_labels

        iter += 1

    return data, new_labels


def main():
    # sample data
    X = multivariate_normal.load_data()

    # kmeans clustering
    k = 2
    Xnew, new_labels = kmeans(X, k)

    # plot
    colors = ['r', 'b']
    for i in range(k):
        plt.scatter(Xnew[new_labels == i, 0],
                    Xnew[new_labels == i, 1],
                    color=colors[i], marker='x')
    plt.show()


if __name__ == '__main__':
    main()
