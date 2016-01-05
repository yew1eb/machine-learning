#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import itertools


def mf_org(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02, threshold=0.001):
    Q = Q.T
    for step in xrange(steps):
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - np.dot(P[i, :], Q[:, j])
                    for k in xrange(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = np.dot(P, Q)
        e = 0
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - np.dot(P[i, :], Q[:, j]), 2)
                    for k in xrange(K):
                        e = e + (beta/2) * (pow(P[i][k], 2) + pow(Q[k][j], 2))
        if e < threshold:
            break
    return P, Q.T


def mf(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02, threshold=0.001):
    pass


def main():
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

    nP_org, nQ_org = mf_org(R, P, Q, K)
    #nP, nQ = mf(R, 2)

    nR_org = np.dot(nP_org, nQ_org.T)
    #nR = np.dot(nP.T, nQ)

    print nR_org

if __name__ == '__main__':
    main()
