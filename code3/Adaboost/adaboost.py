# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 12:42:48 2015

@author: liu
"""
from numpy import *

def loadSimpData():

    dataMat = array([[1.,2.1],[1.5,1.6],[1.3,1.],[1.,1.],[2.,1.],[1.2,1.1],\
    [1.1,0.4],[0.9,1.3],[0.86,1.2],[1.8,1.8],[1.7,1.5],[1.9,1.8]])
    classLabels = array([1.0,1.0,-1.0,-1.0,1.0,-1.0,-1.0,-1.0,\
    -1.0,1.0,1.0,1.0])
    return dataMat,classLabels

# 单树桩分类器，也就是简单的单层决策树弱分类器
# 该函数根据某个最好的特征的最好划分点对数据进行分类
# demen就是特征，threshVal就是划分点，retArray就是返回的分类结果
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
    retArray = ones((shape(dataMatrix)[0],1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen]<=threshVal] = -1.0
    else:
        retArray[dataMatrix[:,dimen]>threshVal] = -1.0
    return retArray
        
# 创建树桩，返回最好的树桩和该树桩的分类最小误差以及分类结果
# 该函数用于从数据集中找到最好的划分特征以及该特征的最好划分点
def buildStump(dataArr,classLabels,D):
    # 建立备份的数据
    dataMatrix = mat(dataArr);
    labelMat = mat(classLabels).T
    # m 是行，也就是每个样本，n是列，也就是每个特征
    m,n = shape(dataMatrix)
    # 步数的设置，也就是在每个特征的最大值和最小值中分几次设置阈值进行数据划分
    # 也就是获取这个特征的最佳划分点，步数越高，特征点寻找得越精细，但耗时更多
    # 用字典来存储bestStump的数据
    numSteps = 10.0;bestStump = {};bestClassEst = mat(zeros((m,1)))
    minError = inf
    # 对每个特征
    for i in range(n):
        # 获取每个特征的最小值和最大值
        rangeMin = dataMatrix[:,i].min();
        rangeMax = dataMatrix[:,i].max();
        stepSize = (rangeMax-rangeMin)/numSteps
        for j in range(-1,int(numSteps)+1):
            for inequal in ['lt','gt']:
                threshVal = (rangeMin + float(j) * stepSize)
                # 预测值
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)
                errArr = mat(ones((m,1)))
                errArr[predictedVals == labelMat] = 0
                # 样本权重乘以样本误差
                weightedError = D.T * errArr
                # 下面的 .2f 表示浮点数小数殿后两位， .3f 表示小数点后3位
                print "split:dim %d,thresh %.2f, thresh ineqal: %s,the weighted eror is %.3f" %\
                (i,threshVal,inequal,weightedError)
                if weightedError < minError:
                    minError = weightedError
                    # 最好的分类结果
                    bestClassEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump,minError,bestClassEst
        
def adaBoostTrainDS(dataArr,classLabels,numIt = 40):
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m,1))/m)
    # 创建矩阵 mat
    aggClassEst = mat(zeros((m,1)))
    for i in range(numIt):
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)
        print 'D:',D.T
        # log 就是 ln 公式： a = 0.5*ln((1-e)/e)
        alpha = float(0.5*log((1.0-error)/max(error,1e-16)))
        bestStump['alpha'] = alpha
        # 将当前的最好的树桩添加到弱分类其数组中
        weakClassArr.append(bestStump)
        print 'classEst:',classEst.T
        # D权重的更新公式，利用原本的类别classLabels与划分的类别classEst做乘积
        # 用来同时计算正确划分和错误划分的公式，也就是自动确定正负号
        expon = multiply(-1*alpha*mat(classLabels).T,classEst)
        # 更新权重D
        D = multiply(D,exp(expon))
        D = D/D.sum()
        aggClassEst += alpha*classEst
        print 'aggClassEst:',aggClassEst.T
        aggErrors = multiply(sign(aggClassEst)!=mat(classLabels).T,ones((m,1)))
        errorRate = aggErrors.sum()/m
        print 'total error:',errorRate,"\n"
        if errorRate == 0.0:break;
    return weakClassArr
    
    # 利用学习得到的多个级联弱分类器进行数据分类
def adaClassify(datToClass,classifierArr):
    dataMatrix = mat(datToClass)
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'],\
        classifierArr[i]['thresh'],classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
        print aggClassEst
    return sign(aggClassEst)
