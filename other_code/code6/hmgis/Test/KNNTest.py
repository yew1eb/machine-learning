# _*_ coding: utf-8 _*_
from sklearn.neighbors import KNeighborsClassifier
from hmgis.Classifier.KNN import *


class KNNTest:
	def _createDataSet(self):
		"""
		返回一个数组
		group为数据
		labels为这些元素的类型
		:return:
		"""
		group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
		labels = ['A', 'A', 'B', 'B']
		return group, labels

	def _loadDataSetFromFile(self, filename):
		"""
		从filename文件获得数据
		:param filename:
		:return:数值数据及其类型
		"""
		fr = open(filename)
		numberOfLines = len(fr.readlines())         #get the number of lines in the file
		returnMat = zeros((numberOfLines, 3))        #prepare matrix to return
		classLabelVector = []                       #prepare labels return
		fr = open(filename)
		index = 0
		for line in fr.readlines():
			line = line.strip()
			listFromLine = line.split('\t')
			returnMat[index, :] = listFromLine[0:3]
			classLabelVector.append(int(listFromLine[-1]))
			index += 1
		return returnMat, classLabelVector

	##---------------------------------------
	def knnTest(self):
		"""
		KNN测试
		"""
		knn = KNNDemo()
		group, labels = self._createDataSet()
		print "原生分类器,[0,1]的分类结果为", knn.predict([0, 1], group, labels, 3)
		## scikit learn代码
		neigh = KNeighborsClassifier(n_neighbors=3)
		neigh.fit(group, labels)
		print "SciKit的KNN分类器,[0,1]的分类结果为", neigh.predict([[0, 1]])

	def show2DPoint(self, filename):
		"""
		点的二维化
		:param filename:
		"""
		datingDataMat, datingLabels = self._loadDataSetFromFile(filename)
		knn = KNNDemo()
		normMat, ranges, minVals = knn.autoNorm(datingDataMat)
		knn.show2D(normMat, datingLabels)

	def show3DPoint(self, filename):
		"""
		点的三维化
		:param filename:
		"""
		datingDataMat, datingLabels = self._loadDataSetFromFile(filename)
		knn = KNNDemo()
		normMat, ranges, minVals = knn.autoNorm(datingDataMat)
		knn.show3D(normMat, datingLabels)

	def knnTest2(self, infile):
		"""
		KNN分类
		:rtype : object
		:param infile:
		"""
		hoRatio = 0.80      #hold out 20%
		datingDataMat, datingLabels = self._loadDataSetFromFile(infile)       #load data setfrom file
		knn = KNNDemo()
		## 归一化处理
		normMat, ranges, minVals = knn.autoNorm(datingDataMat)
		m = normMat.shape[0]
		numTestVecs = int(m * hoRatio)
		errorCount = 0.0
		for i in range(numTestVecs):
			## 第一个参数为要测试的元素，第二个为训练集，第三个为训练集类型，第四个为K参数
			classifierResult = knn.predict(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
			print "分类结果为: %d, 实际结果为: %d" % (classifierResult, datingLabels[i])
			if (classifierResult != datingLabels[i]): errorCount += 1.0
		print "总错误率为: %f" % (errorCount / float(numTestVecs))
		print errorCount, "个错误"


	def knnTestScikit(self, infile):
		"""
		Scikit的KNN分类
		:param infile:
		"""
		hoRatio = 0.80      #hold out 10%
		datingDataMat, datingLabels = self._loadDataSetFromFile(infile)       #load data setfrom file
		knn = KNNDemo()
		normMat, ranges, minVals = knn.autoNorm(datingDataMat)
		m = normMat.shape[0]
		numTestVecs = int(m * hoRatio)
		errorCount = 0.0
		## SCIKIT的KNN分类器
		neigh = KNeighborsClassifier(n_neighbors=3)
		## 训练集
		neigh.fit(normMat[numTestVecs:m, :], datingLabels[numTestVecs:m])
		for i in range(numTestVecs):
			## 分类过程
			classifierResult = neigh.predict(normMat[i, :])
			print "分类结果为: %d, 实际结果为: %d" % (classifierResult, datingLabels[i])
			if (classifierResult != datingLabels[i]): errorCount += 1.0
		print "SCIKIT分类总错误率为: %f" % (errorCount / float(numTestVecs))
		print errorCount