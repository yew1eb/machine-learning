#!/usr/bin/env python
# -*- coding: utf-8 -*-


from sklearn.manifold import MDS
from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt


def plot_embedding(X, y):
    x_min, x_max = np.min(X), np.max(X)
    X = (X - x_min) / (x_max - x_min)

    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.show()


def main():
    digits = load_digits()
    X = digits.data
    y = digits.target
    mds = MDS()
    X_mds = mds.fit_transform(X)
    plot_embedding(X_mds, y)

if __name__ == '__main__':
    main()
