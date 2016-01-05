#!/usr/bin/env python
# -*- coding: utf-8 -*-

import multivariate_normal
from sklearn.cross_validation import train_test_split
import numpy as np
from scipy.optimize import fmin_bfgs
import matplotlib.pyplot as plt


def sigmoid(X):
    OVERFLOW_THRESH = -709
    X = np.sum(X)
    return 0.0 if X < OVERFLOW_THRESH else (1.0 / (1.0 + np.exp(-1.0 * X)))


def compute_cost(theta, X, y):
    X = add_bias(X)
    m = len(X)
    cost = np.sum(
        [- yy * np.log(sigmoid(theta.T * xx) + np.finfo(np.float32).eps)
         - (1-yy) * np.log(1-sigmoid(theta.T * xx) + np.finfo(np.float32).eps)
         for (xx, yy) in zip(X, y)]
    )
    return (1./m)*cost


def compute_grad(theta, X, y):
    X = add_bias(X)
    m, dim = X.shape
    grad = np.zeros([dim, 1])
    for j in range(len(theta)):
        grad[j] = np.sum(
            [(sigmoid(theta.T*xx) - yy)*xx[j] for (xx, yy) in zip(X, y)]
        )
        grad[j] = 1./m
    return grad


def add_bias(X):
    X = np.insert(X, 0, 1, axis=1)
    return X


def main():
    # load sample data
    X, X_label = multivariate_normal.load_data_with_label()

    # split all data into train and test set
    X_train, X_test, X_label_train, X_label_test = train_test_split(X, X_label)

    # compute theta
    m, dim = X.shape
    initial_theta = np.insert(np.zeros([dim, 1], dtype=np.float32),
                              0, 1, axis=0)
    theta = fmin_bfgs(compute_cost, initial_theta, args=(X_train,
                                                         X_label_train))
    print theta

    # plot test data
    colors = ['r', 'b']
    for i in range(2):
        x, y = X_test[X_label_test == i, 0], X_test[X_label_test == i, 1]
        plt.scatter(x, y, color=colors[i], marker='x')

    # plot decision boundary
    # intercept = theta[0]
    a = -1.0 * theta[1] / theta[2]
    xx = np.linspace(-20, 10)
    yy = a * xx - theta[0] / theta[2]
    plt.plot(xx, yy, 'k-')
    plt.show()


if __name__ == '__main__':
    main()
