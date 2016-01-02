# _*_ coding: utf-8 _*_

from numpy import *

## 回归方程
class Regression:
	def standRegres(self, xArr, yArr):
		xMat = mat(xArr);
		yMat = mat(yArr).T
		xTx = xMat.T * xMat
		if linalg.det(xTx) == 0.0:
			print "This matrix is singular, cannot do inverse"
			return
		ws = xTx.I * (xMat.T * yMat)
		return ws

	def lwlr(self, testPoint, xArr, yArr, k=1.0):
		xMat = mat(xArr);
		yMat = mat(yArr).T
		m = shape(xMat)[0]
		weights = mat(eye((m)))
		for j in range(m):                      #next 2 lines create weights matrix
			diffMat = testPoint - xMat[j, :]     #
			weights[j, j] = exp(diffMat * diffMat.T / (-2.0 * k ** 2))
		xTx = xMat.T * (weights * xMat)
		if linalg.det(xTx) == 0.0:
			print "This matrix is singular, cannot do inverse"
			return
		ws = xTx.I * (xMat.T * (weights * yMat))
		#print ws
		return testPoint * ws

	## 获得所有点的估值
	def lwlrTest(self, testArr, xArr, yArr, k=1.0):  #loops over all the data points and applies lwlr to each one
		m = shape(testArr)[0]
		yHat = zeros(m)
		for i in range(m):
			yHat[i] = self.lwlr(testArr[i], xArr, yArr, k)
		return yHat

	def lwlrTestPlot(self, xArr, yArr, k=1.0):  #same thing as lwlrTest except it sorts X first
		yHat = zeros(shape(yArr))       #easier for plotting
		xCopy = mat(xArr)
		xCopy.sort(0)
		for i in range(shape(xArr)[0]):
			yHat[i] = self.lwlr(xCopy[i], xArr, yArr, k)
		return yHat, xCopy

