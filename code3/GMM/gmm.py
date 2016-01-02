# -*- coding: utf-8 -*-

'''
 Description ：GMM in Python
 Author ： LiuLongpo
 Time : 2015年7月26日16:54:48
 Source ：From pluskid
'''

import sys;
sys.path.append("E:\Python\MachineLearning\PythonTools")
import PythonUtils as pu
import matplotlib.pyplot as plt
import numpy as np
'矩阵的逆矩阵需要的库'
from numpy.linalg import *

def gmm(X,K):
    threshold  = 1e-15
    N,D = np.shape(X)
    randV = pu.randIntList(1,N,K)
    centroids = X[randV]
    pMiu,pPi,pSigma = inti_params(centroids,K,X,N,D);
    Lprev = -np.inf
    while True:
        'Estiamtion Step'
        Px = calc_prop(X,N,K,pMiu,pSigma,threshold,D)
        pGamma = Px * np.tile(pPi,(N,1))
        pGamma = pGamma / np.tile((np.sum(pGamma,axis=1)),(K,1)).T
        'Maximization Step'
        Nk = np.sum(pGamma,axis=0)
        pMiu = np.dot(np.dot(np.diag(1 / Nk),pGamma.T),X)
        pPi = Nk / N
        for kk in range(K):
            Xshift = X - np.tile(pMiu[kk],(N,1))
            pSigma[:,:,kk] = (np.dot(np.dot(Xshift.T,np.diag(pGamma[:,kk])),Xshift)) / Nk[kk]

        'check for convergence'            
        L = np.sum(np.log(np.dot(Px,pPi.T)))
        if L-Lprev<threshold:
            break
        Lprev = L

    return Px

def inti_params(centroids,K,X,N,D):
    pMiu = centroids
    pPi = np.zeros((1,K))
    pSigma = np.zeros((D,D,K))
    distmat = np.tile(np.sum(X * X,axis=1),(K,1)).T \
    + np.tile(np.sum(pMiu * pMiu,axis = 1).T,(N,1)) \
    - 2 * np.dot(X,pMiu.T)
    labels = np.argmin(distmat,axis=1)

    for k in range(K):
        Xk = X[labels==k]
        pPi[0][k] = float(np.shape(Xk)[0]) / N # 样本数除以 N 得到概率
        pSigma[:,:,k] = np.cov(Xk.T)
    return pMiu,pPi,pSigma

    '计算概率'
def calc_prop(X,N,K,pMiu,pSigma,threshold,D):
    Px = np.zeros((N,K))
    for k in range(K):
        Xshift = X - np.tile(pMiu[k],(N,1))
        inv_pSigma = inv(pSigma[:,:,k]) \
        + np.diag(np.tile(threshold,(1,np.ndim(pSigma[:,:,k]))))
        tmp = np.sum(np.dot(Xshift,inv_pSigma) * Xshift,axis=1)
        coef = (2*np.pi)**(-D/2) * np.sqrt(np.linalg.det(inv_pSigma))
        Px[:,k] = coef * np.exp(-0.5 * tmp)
    return Px

def test():
    X = pu.readDataFromTxt('testSet.txt')
    num = np.size(X)
    X = np.reshape(X,(num/2,2))
    ppx = gmm(X,4)
    index = np.argmax(ppx,axis=1)
    plt.figure()
    plt.scatter(X[index==0][:,0],X[index==0][:,1],s=60,c=u'r',marker=u'o')
    plt.scatter(X[index==1][:,0],X[index==1][:,1],s=60,c=u'b',marker=u'o')
    plt.scatter(X[index==2][:,0],X[index==2][:,1],s=60,c=u'y',marker=u'o')
    plt.scatter(X[index==3][:,0],X[index==3][:,1],s=60,c=u'g',marker=u'o')


if __name__ == '__main__':
    test()
