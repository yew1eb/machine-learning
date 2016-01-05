# _*_ coding: utf-8 _*_
import numpy as np
import pylab as pl
from sklearn import svm

from hmgis.Classifier.SVM import *


class SVMTest:
	## 加载2维数据
	def loadDataSet(self, fileName):
		dataMat = [];
		labelMat = []
		fr = open(fileName)
		for line in fr.readlines():
			lineArr = line.strip().split('\t')
			dataMat.append([float(lineArr[0]), float(lineArr[1])])
			labelMat.append(float(lineArr[2]))
		return dataMat, labelMat

	## 加载多维数据
	def loadMultiDataSet(self, fileName):
		dataMat = [];
		labelMat = []
		fr = open(fileName)
		## 获得培训集
		dataMat = [];
		labelMat = []
		for line in fr.readlines():
			currLine = line.strip().split('\t')
			lineArr = []
			for i in range(21):
				lineArr.append(float(currLine[i]))
			dataMat.append(lineArr)
			labelMat.append(float(currLine[21]))
		return dataMat, labelMat

	## 加载多维数据
	def loadMultiDataSet2(self, fileName):
		dataMat = [];
		labelMat = []
		fr = open(fileName)
		## 获得培训集
		dataMat = [];
		labelMat = []
		for line in fr.readlines():
			currLine = line.strip().split('\t')
			lineArr = []
			for i in range(2):
				lineArr.append(float(currLine[i]))
			dataMat.append(lineArr)
			labelMat.append(float(currLine[2]))
		return dataMat, labelMat

	def loadImages(self, dirName):
		from os import listdir

		hwLabels = []
		trainingFileList = listdir(dirName)           #load the training set
		m = len(trainingFileList)
		trainingMat = zeros((m, 1024))
		for i in range(m):
			fileNameStr = trainingFileList[i]
			fileStr = fileNameStr.split('.')[0]     #take off .txt
			classNumStr = int(fileStr.split('_')[0])
			if classNumStr == 9:
				hwLabels.append(-1)
			else:
				hwLabels.append(1)
			trainingMat[i, :] = self.img2vector('%s/%s' % (dirName, fileNameStr))
		return trainingMat, hwLabels

	def img2vector(self, filename):
		returnVect = zeros((1, 1024))
		fr = open(filename)
		for i in range(32):
			lineStr = fr.readline()
			for j in range(32):
				returnVect[0, 32 * i + j] = int(lineStr[j])
		return returnVect

	## 线性分类器
	def testLinear(self):
		## 加载数据
		dataArr, labelArr = self.loadDataSet('data/svm/testSet.txt')
		svm = SVMLib()
		## 训练一个线性分类器
		ws, b = svm.fit(dataArr, labelArr, 0.6, 0.001, 40)
		print ws
		dataMat = mat(dataArr)
		## 前半部分计算值为分类结果，后面为实际结果
		## SVM分类器是个二元分类器，其结果为-1或1
		## 因此训练时，训练集的值也为-1或1
		print '-----------------'
		print svm.predict(dataMat[0], ws, b), labelArr[0]

	## 多属性线性分类器
	def testMultiLinear(self):
		## 加载数据
		dataArr, labelArr = self.loadMultiDataSet('data/svm/horseColicTest.txt')

		svm = SVMLib()
		## 训练一个线性分类器
		ws, b = svm.fit(dataArr, labelArr, 0.6, 0.001, 40)
		print ws
		dataMat = mat(dataArr)
		## 前半部分计算值为分类结果，后面为实际结果
		## SVM分类器是个二元分类器，其结果为-1或1
		## 因此训练时，训练集的值也为-1或1
		print '-----------------'
		## 根据SVM判断第4个数据的分类，大于0为1，小于0为-1
		print svm.predict(dataMat[3], ws, b), labelArr[3]


	## rbf核函数分类器
	def testRbf(self, kTup=('rbf', 1.5)):
		dataArr, labelArr = self.loadDataSet('data/svm/testSetRBF.txt')

		svm = SVMLib()
		datMat = mat(dataArr);
		## 训练数据
		RBF = svm.fit_RBF(dataArr, labelArr, 200, 0.0001, 10000, kTup)
		print RBF.b, '----', RBF.alphas
		print "支持向量数量为 %d " % shape(RBF.sVs)[0]
		m, n = shape(datMat)
		errorCount = 0
		for i in range(m):
			## 测试数据
			predict = svm.predict_RBF(RBF, datMat[i, :], kTup)
			if sign(predict) != sign(labelArr[i]): errorCount += 1
		print "培训集错误率: %f" % (float(errorCount) / m)
		#
		dataArr, labelArr = self.loadDataSet('data/svm/testSetRBF2.txt')
		errorCount = 0
		datMat = mat(dataArr);
		m, n = shape(datMat)
		for i in range(m):
			predict = svm.predict_RBF(RBF, datMat[i, :], kTup)
			if sign(predict) != sign(labelArr[i]): errorCount += 1
		print "训练集错误率: %f" % (float(errorCount) / m)

	## rbf核函数分类器
	def testDigits(self, kTup=('rbf', 10)):
		dataArr, labelArr = self.loadImages('data/svm/trainingDigits')

		svm = SVMLib()
		RBF = svm.fit_RBF(dataArr, labelArr, 200, 0.0001, 10000, kTup)
		print RBF.b, '----'
		datMat = mat(dataArr);
		print "there are %d Support Vectors" % shape(RBF.sVs)[0]
		m, n = shape(datMat)
		errorCount = 0
		for i in range(m):
			predict = svm.predict_RBF(RBF, datMat[i, :], kTup)
			if sign(predict) != sign(labelArr[i]): errorCount += 1
		print "the training error rate is: %f" % (float(errorCount) / m)

		dataArr, labelArr = self.loadImages('data/svm/testDigits')
		errorCount = 0
		datMat = mat(dataArr);
		m, n = shape(datMat)
		for i in range(m):
			predict = svm.predict_RBF(RBF, datMat[i, :], kTup)
			if sign(predict) != sign(labelArr[i]): errorCount += 1
		print "the test error rate is: %f" % (float(errorCount) / m)

	def testSciKitSVM(self):
		## 加载数据
		dataArr, labelArr = self.loadMultiDataSet2('data/svm/testSet.txt')
		X = dataArr  # we only take the first two features. We could
		# avoid this ugly slicing by using a two-dim dataset
		Y = labelArr
		X = np.asarray(X)
		Y = np.asarray(Y)
		h = .02  # step size in the mesh

		# we create an instance of SVM and fit out data. We do not scale our
		# data since we want to plot the support vectors
		C = 1.0  # SVM regularization parameter
		svc = svm.SVC(kernel='linear', C=C).fit(X, Y)
		rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, Y)
		poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, Y)
		lin_svc = svm.LinearSVC(C=C).fit(X, Y)

		## 预测过程
		print svc.predict([[7.139979, -2.329896]])

		# create a mesh to plot in
		x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
		y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
		xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
		                     np.arange(y_min, y_max, h))

		# title for the plots
		titles = ['SVC with linear kernel',
		          'SVC with RBF kernel',
		          'SVC with polynomial (degree 3) kernel',
		          'LinearSVC (linear kernel)']

		for i, clf in enumerate((svc, rbf_svc, poly_svc, lin_svc)):
			# Plot the decision boundary. For that, we will assign a color to each
			# point in the mesh [x_min, m_max]x[y_min, y_max].
			pl.subplot(2, 2, i + 1)
			Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
			# Put the result into a color plot
			Z = Z.reshape(xx.shape)
			pl.contourf(xx, yy, Z, cmap=pl.cm.Paired)
			pl.axis('off')
			# Plot also the training points
			pl.scatter(X[:, 0], X[:, 1], c=Y, cmap=pl.cm.Paired)
			pl.title(titles[i])

		pl.show()