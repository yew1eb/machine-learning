# _*_ coding: utf-8 _*_
from hmgis.Classifier.Logistic import *


class LogisticTest:
	def loadDataSet(self, filename):
		dataMat = [];
		labelMat = []
		fr = open(filename)
		for line in fr.readlines():
			lineArr = line.strip().split()
			dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
			labelMat.append(int(lineArr[2]))
		return dataMat, labelMat

	def calGradAscent(self, filename):
		logic = LogisticDemo()
		## 加载数据
		dataArr, labelMat = self.loadDataSet(filename)
		## 根据梯度下降法来计算拟合参数
		weights = logic.fit_gradAscent(dataArr, labelMat)
		logic.plotBestfit(weights, dataArr, labelMat)

	def calRandomGradAscent(self, filename):
		## 根据随机梯度下降法来计算最佳拟合参数，由于只重复200次，因此效果不会比上面的500次更好
		logic = LogisticDemo()
		## 加载数据
		dataArr, labelMat = self.loadDataSet(filename)
		weights = logic.fit_stocGradAscent0(array(dataArr), labelMat)
		logic.plotBestfit(weights, dataArr, labelMat)

	def calRandomGradAscent2(self, filename):
		logic = LogisticDemo()
		## 加载数据
		dataArr, labelMat = self.loadDataSet(filename)
		## 改进随机梯度，同样200次效果更好
		weights = logic.fit_stocGradAscent1(array(dataArr), labelMat, 200)
		logic.plotBestfit(weights, dataArr, labelMat)

	## 加载文件数据成矩阵，进行测试计算
	def colicTest(self):
		frTrain = open('data/logistic/horseColicTraining.txt')
		frTest = open('data/logistic/horseColicTest.txt')
		## 获得培训集
		trainingSet = [];
		trainingLabels = []
		for line in frTrain.readlines():
			currLine = line.strip().split('\t')
			lineArr = []
			for i in range(21):
				lineArr.append(float(currLine[i]))
			trainingSet.append(lineArr)
			trainingLabels.append(float(currLine[21]))
		## 对培训集进行训练
		logic = LogisticDemo()
		trainWeights = logic.fit_stocGradAscent1(array(trainingSet), trainingLabels, 1000)
		errorCount = 0;
		numTestVec = 0.0
		## 读取测试集
		for line in frTest.readlines():
			numTestVec += 1.0
			currLine = line.strip().split('\t')
			lineArr = []
			for i in range(21):
				lineArr.append(float(currLine[i]))
			## 测试函数
			if int(logic.predict(array(lineArr), trainWeights)) != int(currLine[21]):
				errorCount += 1
		errorRate = (float(errorCount) / numTestVec)
		print "the error rate of this test is: %f" % errorRate
		return errorRate

	def multiColicHorseTest(self):
		numTests = 10;
		errorSum = 0.0
		for k in range(numTests):
			errorSum += self.colicTest()
		print "after %d iterations the average error rate is: %f" % (numTests, errorSum / float(numTests))

	def calScikitLogistic(self, filename):
		logic = LogisticDemo()
		## 加载数据
		dataArr, labelMat = self.loadDataSet(filename)
		## 下面是scikit的LogisticRegression函数
		from sklearn import linear_model

		scilogic = linear_model.LogisticRegression()
		## 拟合过程
		scilogic.fit(array(dataArr), labelMat)
		## 预测过程
		## 注意我们故意将数据集变成了三维参数
		print scilogic.predict([1, -1.395634, 4.662541])