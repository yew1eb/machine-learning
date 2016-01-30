#!/usr/local/bin python
# -*- coding: utf-8 -*-

import numpy as np
import multivariate_normal
from math import sqrt
import matplotlib.pyplot as plt


def initialize(data, k):
    # choose one data point as an initial cluster center
    center = [data[0]]

    for i in range(1, k):
        # calculate distances
        dist = np.array([np.min([np.inner(c-d, c-d) for c in center])
                         for d in data])

        # probability distribution
        probs = dist / dist.sum()
        cumprobs = np.cumsum(probs)

        # select next cluster center
        r = np.random.random()
        for j, p in enumerate(cumprobs):
            if r < p:
                i = j
                break
        center.append(data[i])
    return np.array(center)


def euclid_dist(center, data_i):
    return sqrt(sum([
        (center[n] - data_i[n]) ** 2 for n in range(len(center))
    ]))


def assign_label(center, data_i):
    dist = []
    for k in range(len(center)):
        dist.append(euclid_dist(center[k], data_i))
    return np.argsort(np.ravel(dist))[0]


def calculate_center(data_i):
    val = 0.0
    cnt_i = len(data_i)
    for d in data_i:
        val += d
    return (val / cnt_i)


def kmeanspp(data, k, max_iter=300):
    # calculate an initial cluster center
    center = initialize(data, k)

    # assign the cluster labels
    labels = np.zeros(len(data)).astype(np.int)
    for i, d in enumerate(data):
        labels[i] = assign_label(center, d)

    # kmeans clustering
    iter = 0
    while 1:
        # calculate the cluster centers
        for i in range(k):
            center[i] = calculate_center(data[labels == i])

        # assign new labels
        new_labels = np.zeros(len(data)).astype(np.int)
        for i, d in enumerate(data):
            new_labels[i] = assign_label(center, d)

        if np.array_equal(labels, new_labels) or iter > max_iter:
            break
        else:
            labels = new_labels

    return data, new_labels


def main():
    # sample data
    X = multivariate_normal.load_data()

    # kmeans++
    k = 2
    Xnew, new_labels = kmeanspp(X, k)

    # plot
    colors = ['r', 'b']
    for i in range(k):
        plt.scatter(Xnew[new_labels == i, 0],
                    Xnew[new_labels == i, 1],
                    color=colors[i], marker='x')
    plt.show()


if __name__ == '__main__':
    main()
