# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 12:57:27 2015

@author: LiuLongpo
"""
import optunity
import adaboost
import matplotlib.pyplot as plt
from numpy import *
dataMat,classLabels = adaboost.loadSimpData()

#plt.scatter(dataMat[:,0],dataMat[:,1])
# D是样本的权重矩阵
D = mat(ones((5,1))/5)
#adaboost.buildStump(dataMat,classLabels,D)
print 'data train...'
classifierArr = adaboost.adaBoostTrainDS(dataMat,classLabels,30)
print 'getClassifier:',classifierArr
print 'data predict...'
# 学习得到3个分类器，predict时，每一个分类器级联分类得到的预测累加值 
# aggClassEst越来越远离0，也就是正越大或负越大，也就是分类结果越来越强
adaboost.adaClassify([[1,0.8],[1.8,2]],classifierArr)
# 0,lt,1.3   1,lt,1.0   0,lt,0.9
plt.figure()
I = nonzero(classLabels>0)[0]
plt.scatter(dataMat[I,0],dataMat[I,1],s=60,c=u'r',marker=u'o')
I = nonzero(classLabels<0)[0]
plt.scatter(dataMat[I,0],dataMat[I,1],s=60,c=u'b',marker=u'o')


plt.plot([1.32,1.32],[0.5,2.5])
plt.plot([0.5,2.5],[1.42,1.42])
plt.plot([0.97,0.97],[0.5,2.5])

'''
plt.figure()
I = nonzero(classLabels>0)[0]
plt.scatter(dataMat[I,0],dataMat[I,1],s=60,c=u'r',marker=u'o')
I = nonzero(classLabels<0)[0]
plt.scatter(dataMat[I,0],dataMat[I,1],s=60,c=u'b',marker=u'o')
plt.plot([1.32*1.19,1.32*1.19],[0.5,2.5])
plt.plot([0.5,2.5],[1.42*1.52,1.42*1.52])
plt.plot([0.97*1.13,0.97*1.13],[0.5,2.5])
#plt.scatter([0,5],[0,5])

'''
