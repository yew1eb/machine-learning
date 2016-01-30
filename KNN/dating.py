#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Filename: dating.py
@Author: yew1eb
@Date: 2015/12/21 0021
"""
'''
使用k-近邻算法改进约会网站的配对效果,《机器学习实战》
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import operator


def get_data(filename):
    mp = {'didntLike': 1, 'smallDoses': 2, 'largeDoses': 3}
    data = pd.read_table(filename, header=None, names=['x1', 'x2', 'x3', 'label'])
    feature_data = data.loc[:, ['x1', 'x2', 'x3']]
    label_data = data.label.replace(mp.keys(), mp.values())
    return feature_data, label_data


# datingDataMat, datingLabels = get_data("datingTestSet.txt")
# datingDataMat, datingLabels = get_data("testData.txt")
# print(datingDataMat)
# print(datingLabels)

def print_image(datingDataMat, datingLabels):
    fig = plt.figure()
    ax = fig.add_subplot(111)  # 白色背景
    # ax.scatter(datingDataMat['x2'], datingDataMat['x3'])
    ax.scatter(datingDataMat['x2'], datingDataMat['x3'], 15.0 * datingLabels, 15.0 * datingLabels)
    ax.axis([-2, 25, -0.2, 2.0])
    plt.xlabel('Percentage of Time Spent Playing Video Games')
    plt.ylabel('Liters of Ice Cream Consumed Per Week')
    plt.show()


# print_image(datingDataMat,datingLabels)

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# 数据归一化
def auto_norm(data):
    min_value = data.min(0)
    max_value = data.max(0)
    ranges = max_value - min_value
    norm_data = np.zeros(data.shape)
    m = data.shape[0]
    norm_data = data - np.tile(min_value, (m, 1))
    norm_data = norm_data / np.tile(ranges, (m, 1))
    return norm_data, ranges, min_value


# 交叉验证
def datingClassTest():
    hoRatio = 0.30  # 测试数据占的百分比
    datingDataMat, datingLabels = get_data('datingTestSet.txt')
    normMat, ranges, minValues = auto_norm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0  # 错误率
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print('the classifier came back with: %d, the real answer is: %d' % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0

    print("the total error rate is: %f " % (errorCount / float(numTestVecs)))


#输入某人的信息，便得出对对方喜欢程度的预测值
def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = get_data('datingTestSet.txt')
    normMat, ranges, minVals = auto_norm(datingDataMat)
    inArr = np.array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - minVals)/ranges, normMat, datingLabels, 3)
    print('You will probably like this person: ', resultList[classifierResult - 1])

if __name__ == '__main__':
    datingClassTest()