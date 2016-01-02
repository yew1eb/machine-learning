# _*_ coding: utf-8 _*_

from numpy import *


class Bayesian:
	## 寻找出单个文档，其中词汇不重复
	def createVocabList(self, dataSet):
		vocabSet = set([])  #create empty set
		for document in dataSet:
			vocabSet = vocabSet | set(document) #union of the two sets
		return list(vocabSet)

	## 计算一个字符串的单词在词包中的位置
	def setOfWords2Vec(self, vocabList, inputSet):
		returnVec = [0] * len(vocabList)
		for word in inputSet:
			if word in vocabList:
				returnVec[vocabList.index(word)] += 1
			else:
				print "词汇: %s 不在词典中!" % word
		return returnVec

	## 计算先验概率
	## P(B|A) = P(B)P(A|B)/P(B)P(A|B) + P(C)P(A|C)
	def fit(self, trainMatrix, trainCategory):
		numTrainDocs = len(trainMatrix)
		numWords = len(trainMatrix[0])
		## 先验概率
		pAbusive = sum(trainCategory) / float(numTrainDocs)
		p0Num = ones(numWords)
		p1Num = ones(numWords)      #change to ones()
		p0Denom = 2.0;
		p1Denom = 2.0                        #change to 2.0
		## 类型AB的先验概率是根据此类型中词汇在词包中出现的次数除以总次数
		for i in range(numTrainDocs):
			if trainCategory[i] == 1:
				p1Num += trainMatrix[i]
				p1Denom += sum(trainMatrix[i])
			else:
				p0Num += trainMatrix[i]
				p0Denom += sum(trainMatrix[i])
		p1Vect = log(p1Num / p1Denom)         #change to log()
		p0Vect = log(p0Num / p0Denom)          #change to log()
		return p0Vect, p1Vect, pAbusive

	## 比较概率
	def predict(self, vec2Classify, p0Vec, p1Vec, pClass1):
	## 由于所有类型的全概率都是一样的（分母一样），因此可以纯比分子
		p1 = sum(vec2Classify * p1Vec) + log(pClass1)    #element-wise mult
		p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
		if p1 > p0:
			return 1
		else:
			return 0

