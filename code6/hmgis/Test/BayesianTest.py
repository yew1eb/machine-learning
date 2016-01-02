# _*_ coding: utf-8 _*_
from numpy import *
import jieba.posseg as pseg
import jieba
from hmgis.Classifier.Bayesian import *


class BayesianTest:
	def loadDataSet(self):
		postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
		               ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
		               ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
		               ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
		               ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
		               ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
		classVec = [0, 1, 0, 1, 0, 1]    #1 is abusive, 0 not
		return postingList, classVec

	## 测试单个词汇的类型
	def testingNB(self):
	## 加载已有数据集
		listOPosts, listClasses = self.loadDataSet()
		bayesian = Bayesian()
		myVocabList = bayesian.createVocabList(listOPosts)
		trainMat = []
		for postinDoc in listOPosts:
			trainMat.append(bayesian.setOfWords2Vec(myVocabList, postinDoc))
		## 计算已有数据集中的先验概率
		p0V, p1V, pAb = bayesian.fit(array(trainMat), array(listClasses))

		## 测试不同字符串的后验概率
		testEntry = ['love', 'my', 'dalmation']
		thisDoc = array(bayesian.setOfWords2Vec(myVocabList, testEntry))
		print testEntry, '被分类为: ', bayesian.predict(thisDoc, p0V, p1V, pAb)
		testEntry = ['stupid', 'garbage']
		thisDoc = array(bayesian.setOfWords2Vec(myVocabList, testEntry))
		print testEntry, '被分类为: ', bayesian.predict(thisDoc, p0V, p1V, pAb)


class RSSBayesianTest:
	## 解析RSS源，返回实际的RSS数量
	def loadRSS(self, webpage, file):
		#OUT = "data/bayesian/rss/rss_junshi.txt"
		f = open(file, "w")

		import feedparser
		#mil = feedparser.parse('http://mil.sohu.com/rss/junshi.xml')
		mil = feedparser.parse(webpage)
		count = 0
		for entry in mil["entries"]:
			title = entry["title"]
			title = title.replace('\n', '')
			txt = entry["summary_detail"]["value"]
			txt = txt.replace('\n', '')
			if len(txt) > 1:
				f.writelines(title + '。' + txt + "\n")
				count = count + 1
		f.close()
		return count

	## 预处理从RSS源获得的数据
	def preProcess(self, filepath, classtag):

		sentences = [line for line in file(filepath) if len(line) > 1]
		jieba.load_userdict("lib/userdict")
		## 使用jieba对文本进行分词
		texts_tokenized = [
			[word for word in pseg.cut(document) if word.flag in ['n', 'nr', 'ns', 'nt', 'nz', 'nl', 'ng', 'eng', 'x']]
			for document in sentences]
		## 停用词
		chnstopwordoc = [line.strip() for line in file('lib/chinesestopwords.txt')]
		stoplist = [course for course in chnstopwordoc]
		## 剔除停用词
		texts_tokenized = [[word for word in document if not word.word in stoplist] for document in texts_tokenized]
		## 剔除长度为1的词汇
		wordlist = [[word.word for word in document if len(word.word) > 1] for document in texts_tokenized]
		print len(wordlist)
		labels = []
		for i in range(0, len(wordlist)):
			labels.append(classtag)

		return wordlist, labels

	def getData(self, filepath):
		sentences = [line for line in file(filepath)]
		return sentences

	def testEntryProcess(self, text):
		## 使用jieba对文本进行分词
		texts_tokenized = [word for word in pseg.cut(text) if
		                   word.flag in ['n', 'nr', 'ns', 'nt', 'nz', 'nl', 'ng', 'eng', 'x']]
		## 停用词
		chnstopwordoc = [line.strip() for line in file('lib/chinesestopwords.txt')]
		stoplist = [course for course in chnstopwordoc]
		## 剔除停用词
		texts_tokenized = [t.word for t in texts_tokenized if not t.word in stoplist]
		## 剔除长度为1的词汇
		wordlist = [document for document in texts_tokenized if len(document) > 1]
		return wordlist

	def loadProcessedData(self):
		dataMat0, labels0 = self.preProcess('data/bayesian/rss/rss_junshi.txt', 0)
		dataMat1, labels1 = self.preProcess('data/bayesian/rss/rss_tiyu.txt', 1)
		dataMat = dataMat0 + dataMat1
		labels = labels0 + labels1
		return dataMat, labels


	def SingleClassifier(self):
		## 加载RSS源并将其保存为文本文件
		## 除非是生成新数据，否则不执行这段代码
		#juns_count = rss.loadRSS('http://mil.sohu.com/rss/junshi.xml','data/bayesian/rss/rss_junshi.txt')
		#tiyu_count = rss.loadRSS('http://rss.news.sohu.com/rss/sports.xml','data/bayesian/rss/rss_tiyu.txt' )
		#print juns_count
		#print tiyu_count

		dataMat, labels = self.loadProcessedData()

		bayesian = Bayesian()
		myVocabList = bayesian.createVocabList(dataMat)
		## 建立bag of words 矩阵
		trainMat = []
		for postinDoc in dataMat:
			trainMat.append(bayesian.setOfWords2Vec(myVocabList, postinDoc))
		## 计算已有数据集中的先验概率
		p0V, p1V, pAb = bayesian.fit(array(trainMat), array(labels))

		## 测试不同字符串的后验概率
		testText = "美国军队的军舰今天访问了巴西港口城市，并首次展示了核潜艇攻击能力，飞机，监听。他们表演了足球。"
		testEntry = self.testEntryProcess(testText)
		thisDoc = array(bayesian.setOfWords2Vec(myVocabList, testEntry))
		clabels = ['军事', '体育']
		print testText, 'classified as: ', clabels[bayesian.predict(thisDoc, p0V, p1V, pAb)]


	## 交叉分类验证
	## 从51个样本中选出41个培训集，10个测试集
	def crossValidClassifier(self):
		dataMat, labels = self.loadProcessedData()
		bayesian = Bayesian()
		myVocabList = bayesian.createVocabList(dataMat)
		trainingSet = range(51);
		testSet = []           #create test set
		for i in range(10):
			randIndex = int(random.uniform(0, len(trainingSet)))
			testSet.append(trainingSet[randIndex])
			del (trainingSet[randIndex])
		trainMat = [];
		trainClasses = []
		for docIndex in trainingSet:#train the classifier (get probs) trainNB0
			trainMat.append(bayesian.setOfWords2Vec(myVocabList, dataMat[docIndex]))
			trainClasses.append(labels[docIndex])
		p0V, p1V, pSpam = bayesian.fit(array(trainMat), array(trainClasses))

		clabels = ['军事', '体育']
		data = self.getData('data/bayesian/rss/rss_junshi.txt') + self.getData('data/bayesian/rss/rss_tiyu.txt')
		errorCount = 0
		for docIndex in testSet:        #classify the remaining items
			wordVector = bayesian.setOfWords2Vec(myVocabList, dataMat[docIndex])
			type = bayesian.predict(array(wordVector), p0V, p1V, pSpam)
			if type != labels[docIndex]:
				errorCount += 1
				print "判断类型：", clabels[type]
				print "classification error", data[docIndex]
				print "---------------------------------------"
		print 'the error rate is: ', float(errorCount) / len(testSet)

	## 使用scikti代码进行GaussianNB训练
	def scikitNBClassfier(self):
		dataMat, labels = self.loadProcessedData()
		bayesian = Bayesian()
		myVocabList = bayesian.createVocabList(dataMat)
		## 建立bag of words 矩阵
		trainMat = []
		for postinDoc in dataMat:
			trainMat.append(bayesian.setOfWords2Vec(myVocabList, postinDoc))

		from sklearn.naive_bayes import GaussianNB

		gnb = GaussianNB()
		X = array(trainMat)
		y = labels

		testText = "美国军队的军舰今天访问了巴西港口城市，并首次展示了核潜艇攻击能力，飞机，监听。他们表演了足球。"
		testEntry = self.testEntryProcess(testText)

		bayesian = Bayesian()
		thisDoc = array(bayesian.setOfWords2Vec(myVocabList, testEntry))
		## 拟合并预测
		y_pred = gnb.fit(X, y).predict(thisDoc)
		clabels = ['军事', '体育']
		y_pred = gnb.fit(X, y).predict(X)
		print("Number of mislabeled points : %d" % (labels != y_pred).sum())

## 电子邮件测试器
class emailClassfier:
	def textParse(self, bigString):
		import re

		listOfTokens = re.split(r'\W*', bigString)
		return [tok.lower() for tok in listOfTokens if len(tok) > 2]

	def spamTest(self, bayesian):
		docList = [];
		classList = [];
		fullText = []
		for i in range(1, 26):
			wordList = self.textParse(open('data/bayesian/email/spam/%d.txt' % i).read())
			docList.append(wordList)
			fullText.extend(wordList)
			classList.append(1)
			wordList = self.textParse(open('data/bayesian/email/ham/%d.txt' % i).read())
			docList.append(wordList)
			fullText.extend(wordList)
			classList.append(0)

		bayesian = Bayesian()
		vocabList = bayesian.createVocabList(docList)#create vocabulary
		trainingSet = range(50);
		testSet = []           #create test set
		for i in range(10):
			randIndex = int(random.uniform(0, len(trainingSet)))
			testSet.append(trainingSet[randIndex])
			del (trainingSet[randIndex])
		trainMat = [];
		trainClasses = []
		for docIndex in trainingSet:#train the classifier (get probs) trainNB0
			trainMat.append(bayesian.setOfWords2Vec(vocabList, docList[docIndex]))
			trainClasses.append(classList[docIndex])
		p0V, p1V, pSpam = bayesian.fit(array(trainMat), array(trainClasses))
		errorCount = 0
		for docIndex in testSet:        #classify the remaining items
			wordVector = bayesian.setOfWords2Vec(vocabList, docList[docIndex])
			if bayesian.predict(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
				errorCount += 1
				print "分类错误", docList[docIndex]
		print '错误率是: ', float(errorCount) / len(testSet)
		#return vocabList,fullText