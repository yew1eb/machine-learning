#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import itertools


def mf(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02, threshold=0.001):
    # Q = M x K -> Q.T = K x M
    Q = Q.T

    rows, cols = len(R), len(R[0])
    while steps > 0:
        # gradient descent
        for i, j in itertools.product(xrange(rows), xrange(cols)):
            if R[i][j] > 0:
                eij = R[i][j] - np.dot(P[i, :], Q[:, j])
                for k in xrange(K):
                    P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                    Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])

        # update
        eR = np.dot(P, Q)
        e = 0
        for i, j in itertools.product(xrange(rows), xrange(cols)):
            if R[i][j] > 0:
                e = e + pow(R[i][j] - np.dot(P[i, :], Q[:, j]), 2)
                for k in xrange(K):
                    e = e + pow(R[i][j] - np.dot(P[i, :], Q[:, j]), 2) + \
                        (beta/2) * (pow(P[i][k], 2) + pow(Q[k][j], 2))
        if e < threshold:
            break

        # decrement steps
        steps -= 1

    return P, Q.T


def main():
    # create sample data
    R = np.array([
        [5, 3, 0, 1],
        [4, 0, 0, 1],
        [1, 1, 0, 5],
        [1, 0, 0, 4],
        [0, 1, 5, 4],
    ])

    N = len(R)
    M = len(R[0])
    K = 2

    P = np.random.rand(N, K)
    Q = np.random.rand(M, K)

    # matrix factorization
    nP, nQ = mf(R, P, Q, K)
    nR = np.dot(nP, nQ.T)
    print nR


if __name__ == '__main__':
    main()
