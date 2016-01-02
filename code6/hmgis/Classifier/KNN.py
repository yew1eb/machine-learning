#coding:utf-8
#采用原生代码和scikit Learn库共同测试

import operator
from numpy import *
import matplotlib.pyplot as plt
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D


class KNNDemo:
	def predict(self, inX, dataSet, labels, k):
		"""
		分类器
		:param inX: 需要分类的数据
		:param dataSet: 训练集
		:param labels: 训练集分类
		:param k: KNN选择的数量
		:return:
		"""
		dataSetSize = dataSet.shape[0]
		diffMat = tile(inX, (dataSetSize, 1)) - dataSet
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

	def autoNorm(self, dataSet):
		"""
		归一化处理函数，以防止参数之间的值差距过大
		使得矩阵所有行的所有值均在-1至1之间
		:param dataSet:
		:return:
		"""
		minVals = dataSet.min(0)
		maxVals = dataSet.max(0)
		ranges = maxVals - minVals
		normDataSet = zeros(shape(dataSet))
		m = dataSet.shape[0]
		normDataSet = dataSet - tile(minVals, (m, 1))
		normDataSet = normDataSet / tile(ranges, (m, 1))   #element wise divide
		return normDataSet, ranges, minVals

	def show2D(self, datingDataMat, datingLabels):
		"""
		将点数据进行2维可视化
		:param datingDataMat:
		:param datingLabels:
		"""
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
		plt.show()

	def show3D(self, datingDataMat, datingLabels):
		"""
		将点数据进行三维可视化，即最多只能显示3个属性
		:param datingDataMat:
		:param datingLabels:
		"""
		fig = pl.figure(1, figsize=(8, 6))
		ax = Axes3D(fig, elev=-150, azim=110)
		ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], datingDataMat[:, 0], c=datingLabels)
		ax.set_title("Point 3 Properties Visualization")
		ax.set_xlabel("1st")
		ax.set_ylabel("2nd")
		ax.set_zlabel("3rd")
		pl.show()
