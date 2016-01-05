# _*_ coding: utf-8 _*_

from numpy import *


class AdaBoost:
	def stumpClassify(self, dataMatrix, dimen, threshVal, threshIneq):
		#just classify the data
		"""

		:param dataMatrix:
		:param dimen:
		:param threshVal:
		:param threshIneq:
		:return:
		"""
		retArray = ones((shape(dataMatrix)[0], 1))
		if threshIneq == 'lt':
			retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
		else:
			retArray[dataMatrix[:, dimen] > threshVal] = -1.0
		return retArray


	def buildStump(self, dataArr, classLabels, D):
		"""

		:param dataArr:
		:param classLabels:
		:param D:
		:return:
		"""
		dataMatrix = mat(dataArr);
		labelMat = mat(classLabels).T
		m, n = shape(dataMatrix)
		numSteps = 10.0;
		bestStump = {};
		bestClasEst = mat(zeros((m, 1)))
		minError = inf #init error sum, to +infinity
		for i in range(n):#loop over all dimensions
			rangeMin = dataMatrix[:, i].min();
			rangeMax = dataMatrix[:, i].max();
			stepSize = (rangeMax - rangeMin) / numSteps
			for j in range(-1, int(numSteps) + 1):#loop over all range in current dimension
				for inequal in ['lt', 'gt']: #go over less than and greater than
					threshVal = (rangeMin + float(j) * stepSize)
					predictedVals = self.stumpClassify(dataMatrix, i, threshVal,
					                                   inequal)#call stump classify with i, j, lessThan
					errArr = mat(ones((m, 1)))
					errArr[predictedVals == labelMat] = 0
					weightedError = D.T * errArr  #calc total error multiplied by D
					#print "split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError)
					if weightedError < minError:
						minError = weightedError
						bestClasEst = predictedVals.copy()
						bestStump['dim'] = i
						bestStump['thresh'] = threshVal
						bestStump['ineq'] = inequal
		return bestStump, minError, bestClasEst


	def fit(self, dataArr, classLabels, numIt=40):
		"""

		:rtype : object
		:param dataArr: 培训数据集
		:param classLabels: 培训数据标签
		:param numIt: 迭代次数
		:return:
		"""
		weakClassArr = []
		m = shape(dataArr)[0]
		D = mat(ones((m, 1)) / m)   #init D to all equal
		aggClassEst = mat(zeros((m, 1)))
		for i in range(numIt):
			bestStump, error, classEst = self.buildStump(dataArr, classLabels, D)#build Stump
			#print "D:",D.T
			alpha = float(
				0.5 * log((1.0 - error) / max(error, 1e-16)))#calc alpha, throw in max(error,eps) to account for error=0
			bestStump['alpha'] = alpha
			weakClassArr.append(bestStump)                  #store Stump Params in Array
			#print "classEst: ",classEst.T
			expon = multiply(-1 * alpha * mat(classLabels).T, classEst) #exponent for D calc, getting messy
			D = multiply(D, exp(expon))                              #Calc New D for next iteration
			D = D / D.sum()
			#calc training error of all classifiers, if this is 0 quit for loop early (use break)
			aggClassEst += alpha * classEst
			#print "aggClassEst: ",aggClassEst.T
			aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m, 1)))
			errorRate = aggErrors.sum() / m
			print "total error: ", errorRate
			if errorRate == 0.0: break

		return weakClassArr, aggClassEst


	def predict(self, datToClass, classifierArr):
		"""

		:param datToClass:
		:param classifierArr:
		:return:
		"""
		dataMatrix = mat(datToClass)#do stuff similar to last aggClassEst in adaBoostTrainDS
		m = shape(dataMatrix)[0]
		aggClassEst = mat(zeros((m, 1)))
		for i in range(len(classifierArr)):
			classEst = self.stumpClassify(dataMatrix, classifierArr[i]['dim'], classifierArr[i]['thresh'],
			                              classifierArr[i]['ineq'])
			aggClassEst += classifierArr[i]['alpha'] * classEst
			print aggClassEst
		return sign(aggClassEst)

	def plotROC(self, predStrengths, classLabels):
		import matplotlib.pyplot as plt

		cur = (1.0, 1.0) #cursor
		ySum = 0.0 #variable to calculate AUC
		numPosClas = sum(array(classLabels) == 1.0)
		yStep = 1 / float(numPosClas);
		xStep = 1 / float(len(classLabels) - numPosClas)
		sortedIndicies = predStrengths.argsort()#get sorted index, it's reverse
		fig = plt.figure()
		fig.clf()
		ax = plt.subplot(111)
		#loop through all the values, drawing a line segment at each point
		for index in sortedIndicies.tolist()[0]:
			if classLabels[index] == 1.0:
				delX = 0;
				delY = yStep;
			else:
				delX = xStep;
				delY = 0;
				ySum += cur[1]
			#draw line from cur to (cur[0]-delX,cur[1]-delY)
			ax.plot([cur[0], cur[0] - delX], [cur[1], cur[1] - delY], c='b')
			cur = (cur[0] - delX, cur[1] - delY)
		ax.plot([0, 1], [0, 1], 'b--')
		plt.xlabel('False positive rate');
		plt.ylabel('True positive rate')
		plt.title('ROC curve for AdaBoost horse colic detection system')
		ax.axis([0, 1, 0, 1])
		plt.show()
		print "the Area Under the Curve is: ", ySum * xStep