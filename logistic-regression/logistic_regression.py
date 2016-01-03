#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Filename: logistic_ression.py
@Author: yew1eb
@Date: 2015/12/20 0020
"""
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def gradient(data, w, j):

def Logistic_Regression(data, listA, listW, listLostFunction):
    N = len(data[0]) #维度
    W = np.zeros(N)
    WNew = np.zeros(N)
    g = np.zeros(N)

    alpha = 100.0 # 学习率随意初始化
    for times in range(10000):



if __name__ == '__main__':
    data = pd.read_csv("logistic_data.txt", sep='\t')

    listA = [] # 每一步的学习率
    listW = [] # 每一步的权值
    listLostFunction = [] # 每一步的损失函数的值
    w = Logistic_Regression(data, listA, listW,listLostFunction)

    # 绘制学习率
    plt.plot(listA, 'r-', linewidth=2)
    plt.plot(listA, 'go')
    plt.xlabel('Times')
    plt.ylabel('Ratio/Step')
    plt.grid(True)
    plt.show()

    # 绘制损失
    listLostFunction.pop(0)
    plt.plot(listLostFunction, 'r-', linewidth=2)
    plt.plot(listLostFunction, 'go')
    plt.xlabel('Times')
    plt.ylabel('Loss Value')
    plt.grid(True)
    plt.show()

    # 绘制权值
    X = []
    Y = []
    for d in data:
        X.append(d[0])
        Y.append(d[1])

    plt.plot(X, Y, 'go', label=u'Original Data', alpha=0.75)
    plt.grid(True)
    x = [min(X), max(X)]
    y = [w[0] * x[0] + w[1], w[0] * x[1] + w[1]]
    plt.plot(x, y, 'r-', linewidth=3, label='Regression Curve')
    plt.legend(loc='upper left')
    plt.show()