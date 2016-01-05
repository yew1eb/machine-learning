# _*_ coding: utf-8 _*_
from numpy import *
from hmgis.Classifier.AdaBoost import *

## AdaBoost是采用Boosting方法来将弱分类器（比随机比例50%好不了多少）
## 通过组合的方式来构成强分类器，提升分类水平
class AdaBoostTest:
	def loadSimpData(self):
		datMat = matrix([[1., 2.1],
		                 [2., 1.1],
		                 [1.3, 1.],
		                 [1., 1.],
		                 [2., 1.]])
		classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
		return datMat, classLabels

	def loadDataSet(self, fileName):
		"""
		修改后的数据加载器，能自动从数据中获得一个表
		:param fileName:
		:return:
		"""
		numFeat = len(open(fileName).readline().split('\t')) #get number of fields
		dataMat = []
		labelMat = []
		fr = open(fileName)
		for line in fr.readlines():
			lineArr = []
			curLine = line.strip().split('\t')
			for i in range(numFeat - 1):
				lineArr.append(float(curLine[i]))
			dataMat.append(lineArr)
			labelMat.append(float(curLine[-1]))
		return dataMat, labelMat


	def adaboostTest(self):
		"""


		"""
		dataArr, labels = self.loadSimpData()

		ada = AdaBoost()
		## dataArr 数据集,labels 数据类型, 30 迭代次数
		classifierArr, aggClassEst = ada.fit(dataArr, labels, 30)
		# 对[0,0]的分类
		print ada.predict([0, 0], classifierArr)

		## 训练数据对数据进行分类
		dataArr, labels = self.loadDataSet("data/adaboost/horseColicTraining2.txt")
		## 培训数据
		classifierArr, aggClassEst = ada.fit(dataArr, labels, 50)
		testdataArr, testlabels = self.loadDataSet("data/adaboost/horseColicTest2.txt")
		## 数据分类
		prediction10 = ada.predict(testdataArr, classifierArr)
		errArr = mat(ones((67, 1)))
		print errArr[prediction10 != mat(testlabels).T].sum() / 67
		## 展示ROC曲线
		ada.plotROC(aggClassEst.T, labels)


	def scikitAdaboost(self):
		"""


		"""
		import numpy as np
		from sklearn.ensemble import AdaBoostClassifier
		from sklearn.tree import DecisionTreeClassifier

		X, y = self.loadDataSet("data/adaboost/horseColicTraining2.txt")
		X = np.asarray(X)
		y = np.asarray(y)
		# Create and fit an AdaBoosted decision tree
		bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200)
		bdt.fit(X, y)
		testdataArr, testlabels = self.loadDataSet("data/adaboost/horseColicTest2.txt")
		Z = bdt.predict(np.asarray(testdataArr))
		print Z
		print mat(testlabels)
