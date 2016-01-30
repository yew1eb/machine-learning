# _*_ coding: utf-8 _*_
from numpy import *
from hmgis.Classifier.Regression import *


class RegressionTest:
	def loadDataSet(self, fileName):      #general function to parse tab -delimited floats
		numFeat = len(open(fileName).readline().split('\t')) - 1 #get number of fields
		dataMat = [];
		labelMat = []
		fr = open(fileName)
		for line in fr.readlines():
			lineArr = []
			curLine = line.strip().split('\t')
			for i in range(numFeat):
				lineArr.append(float(curLine[i]))
			dataMat.append(lineArr)
			labelMat.append(float(curLine[-1]))
		return dataMat, labelMat

	## 一元线性回归方程
	def simpleTest(self):
		xArr, yArr = self.loadDataSet('data/reg/ex0.txt')
		print xArr[0:2]

		reg = Regression()
		ws = reg.standRegres(xArr, yArr)
		print ws
		xMat = mat(xArr);
		yMat = mat(yArr)
		yHat = xMat * ws

		import matplotlib.pyplot as plt

		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])
		xCopy = xMat.copy()
		xCopy.sort(0)
		yHat = xCopy * ws
		m = xCopy[:, 1]
		ax.plot(array(m), array(yHat))
		plt.show()

	## 多元回归
	def multiTest(self):
		xArr, yArr = self.loadDataSet('data/reg/ex0.txt')

		reg = Regression()
		yHat = reg.lwlrTest(xArr, xArr, yArr, 0.01)
		xMat = mat(xArr)
		strInd = xMat[:, 1].argsort(0)
		xSort = xMat[strInd][:, 0, :]

		import matplotlib.pyplot as plt

		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.scatter(xMat[:, 1].flatten().A[0], mat(yArr).T[:, 0].flatten().A[0])
		ax.plot(array(xSort[:, 1]), array(yHat[strInd]))
		plt.show()

	## 接下来的都用scikit的函数算了
	def ridgeRegression(self):
		xArr, yArr = self.loadDataSet('data/reg/abalone.txt')

		## 岭回归
		from sklearn import linear_model

		clf = linear_model.Ridge(alpha=.5)
		clf.fit(xArr, yArr)
		print clf.coef_
		print clf.intercept_
		print '------------------------------------'
		## 预测过程
		print clf.predict([[1, 0.35, 0.265, 0.09, 0.2255, 0.0995, 0.0485, 0.07]])

		## CART决策树回归
		from sklearn.tree import DecisionTreeRegressor

		clf_2 = DecisionTreeRegressor(max_depth=5)
		clf_2.fit(xArr, yArr)
		print clf_2.predict([[1, 0.35, 0.265, 0.09, 0.2255, 0.0995, 0.0485, 0.07]])
