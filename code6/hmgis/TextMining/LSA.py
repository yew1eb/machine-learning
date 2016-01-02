# _*_ coding: utf-8 _*_

from math import log

from numpy import *
from numpy import linalg as la
from numpy import asarray, sum


class LSA(object):
	def __init__(self, stopwords, ignorechars):
		self.stopwords = stopwords
		self.ignorechars = ignorechars
		self.wdict = {}
		self.dcount = 0

	def parse(self, doc):
		"""
		逐行解析英文文件中的字符串，最后生成一个词典【单词, 出现的文档list】
		如[vesting, [1,2,3]
		:param doc:
		"""
		words = doc.split()
		for w in words:
			w = w.lower().translate(None, self.ignorechars)
			if w in self.stopwords:
				continue
			elif w in self.wdict:
				self.wdict[w].append(self.dcount)
			else:
				self.wdict[w] = [self.dcount]
		self.dcount += 1

	def parseEnglish(self, doc):
		"""
		逐行解析英文文件中的字符串，最后生成一个词典【单词, 出现的文档list】
		如[vesting, [1,2,3]
		:param doc:
		"""
		words = doc.split()
		for w in words:
			w = w.lower().translate(None, self.ignorechars)
			if w in self.stopwords:
				continue
			elif w in self.wdict:
				self.wdict[w].append(self.dcount)
			else:
				self.wdict[w] = [self.dcount]
		self.dcount += 1

	def parseChinese(self, infile):
		"""
		逐行解析中文文件中的字符串，最后生成一个词典【单词, 出现的文档list】
		如[vesting, [1,2,3]
		:param doc:
		"""
		texts = [line.strip() for line in file(infile)]
		for text in texts:
			for w in text.split('\t'):
				if w in self.stopwords:
					continue
				elif w in self.wdict:
					self.wdict[w].append(self.dcount)
				else:
					self.wdict[w] = [self.dcount]
			self.dcount += 1


	## 生成一个DTM矩阵
	def buildTDM(self):
		## 将只出现过1次的词汇剔除,keys中存储出现过两次及以上的词汇
		self.keys = [k for k in self.wdict.keys() if len(self.wdict[k]) > 1]
		self.keys.sort()
		self.A = zeros([len(self.keys), self.dcount])
		for i, k in enumerate(self.keys):
			for d in self.wdict[k]:
				self.A[i, d] += 1
		## 最后输出的是一个词汇-文档矩阵，即行为词汇，列为文档
		self.A = self.A.T

	def buildDTM(self):
		"""
		最后输出的是一个文档-词汇矩阵，即行为文档，列为词汇
		"""
		## 将只出现过1次的词汇剔除,keys中存储出现过两次及以上的词汇
		self.keys = [k for k in self.wdict.keys() if len(self.wdict[k]) > 1]
		self.keys.sort()
		self.A = zeros([len(self.keys), self.dcount])
		for i, k in enumerate(self.keys):
			for d in self.wdict[k]:
				self.A[i, d] += 1


	## 建构TF-IDF矩阵
	def TFIDF(self):
		WordsPerDoc = sum(self.A, axis=0)
		DocsPerWord = sum(asarray(self.A > 0, 'i'), axis=1)
		docsperword = 0
		rows, cols = self.A.shape
		for i in range(rows):
			for j in range(cols):
				docsperword = DocsPerWord[i]
				if docsperword == 0:
					docsperword = 1
				self.A[i, j] = (self.A[i, j] / WordsPerDoc[j]) * log(float(cols) / docsperword)

	def printA(self):
		print '矩阵尺寸为', self.A.shape
		print self.A

	def calc(self):
		self.U, self.S, self.Vt = la.svd(self.A)
		return self.U, self.S, self.Vt

	def maxWeight(self, Sigma, weight=0.9):
		sig2 = Sigma ** 2
		sigsum = sum(sig2)
		sigsum_value = sigsum * weight
		for n in range(len(Sigma)):
			if sum(sig2[:n]) > sigsum_value:
				return n

