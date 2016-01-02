# _*_ coding: utf-8 _*_
from numpy import *
from scipy.stats import *
from hmgis.TextMining.LSA import *


class LSATest:
	def __init__(self):
		self.STOP_WORDS_SET = []
		self.PUNCTUATION = ''',:'!()&-?/.";'''

	def simpleTest(self):
		"""
		LSA最简单的例子，这里的文本数据都保存在一个list中
		"""
		docs = [
			"The Neatest Little Guide to Stock Market Investing",
			"Investing For Dummies, 4th Edition",
			"The Little Book of Common Sense Investing: The Only Way to Guarantee Your Fair Share of Stock Market Returns",
			"The Little Book of Value Investing",
			"Value Investing: From Graham to Buffett and Beyond",
			"Rich Dad's Guide to Investing: What the Rich Invest in, That the Poor and the Middle Class Do Not!",
			"Investing in Real Estate, 5th Edition",
			"Stock Investing For Dummies",
			"Rich Dad's Advisors: The ABC's of Real Estate Investing: The Secrets of Finding Hidden Profits Most Investors Miss"]

		stopwordsfile = open('lib/stopwords.txt', "r")
		for word in stopwordsfile: # a stop word in each line
			word = word.replace("\n", '')
			word = word.replace("\r\n", '')
			self.STOP_WORDS_SET.append(word)

		lsa = LSA(self.STOP_WORDS_SET, self.PUNCTUATION)
		## 解析英文文档，注意这里9个文档中每个文档都是一个字符串
		for t in docs:
			lsa.parseEnglish(t)
		## 生成一个TDM矩阵,行为词汇，列为文档
		lsa.buildTDM()
		## 输出词频权重
		lsa.printA()
		lsa.TFIDF()
		## 输出TF-IDF权重
		lsa.printA()
		print '计算两个词汇estate与market的Spearman Rank Corralation相关性'
		w1 = lsa.A[3]
		w2 = lsa.A[6]
		print spearmanr(w1, w2)[0]
		print '----------------------'
		## 计算SVD
		u, s, vt = lsa.calc()

		print "奇异值矩阵为"
		print s
		n90 = lsa.maxWeight(s, 0.9)
		print "前项目占据了奇异值信息量的90%:", n90

		Sig3 = mat([[s[0], 0, 0, 0, 0], [0, s[1], 0, 0, 0], [0, 0, s[2], 0, 0], [0, 0, 0, s[3], 0], [0, 0, 0, 0, s[4]]])
		#print Sig3
		print "模拟原始矩阵"
		newMat = mat(u[:, 0:5]) * Sig3 * mat(vt[0:5, :])
		print newMat
		print '两个矩阵之差'
		t = mat(lsa.A) - newMat
		print t
		w1 = asarray(newMat[3, :])[0]
		w2 = asarray(newMat[6, :])[0]
		print spearmanr(w1, w2)[0]
		print "对比w1和w2的前后spearmanr系数，可知其相关性提升了"

	## 这个的语料库为英文，保存在一个文件中，每行为一个
	def englishCorpusTest(self):
		courses = [line.strip() for line in file('data/LSA/coursera_corpus')]
		courses_name = [course.split('\t')[0] for course in courses]
		print courses_name[0:2]

		stopwords = ['and', 'edition', 'for', 'in', 'little', 'of', 'the', 'to', 'a', '1', '2', '3', '4']
		ignorechars = ''',:'!()&-'''
		lsa = LSA(stopwords, ignorechars)

		for t in courses_name:
			lsa.parseEnglish(t)
		lsa.buildTDM()
		## 输出词频权重
		lsa.printA()
		lsa.TFIDF()
		## 输出TF-IDF权重
		lsa.printA()
		## 计算SVD
		u, s, vt = lsa.calc()
		print "奇异值矩阵为"
		print u.shape, s.shape, vt.shape
		print s
		n90 = lsa.maxWeight(s, 0.9)
		print "前项目占据了奇异值信息量的90%:", n90

		emptyMat = zeros(shape=(n90, n90))
		i = 0
		for a in emptyMat:
			a[i] = s[i]
			i = i + 1

	## 中文处理，所有的中文语料库都进行了分词整理，并保存在一个文件中，每行数据表示一个文档
	def weiboTest(self):
		# 以下三行代码是将数据从csv转化为txt
		#from hmgis.TextMining.TextTools.parseFile import *
		#p = parseCSV('lib/chinesestopwords.txt', 'lib/userdict')
		#p.parse('data/LSA/jiangbt.csv', 'data/LSA/wb_clean.txt')

		lsa = LSA([], [])
		lsa.parseChinese('data/LSA/wb_clean.txt')

		lsa.buildTDM()
		lsa.printA()
		lsa.TFIDF()
		lsa.printA()
		u, s, vt = lsa.calc()
		print "奇异值矩阵为"
		print u.shape, s.shape, vt.shape
		print s
		n90 = lsa.maxWeight(s, 0.4)
		print "前项目占据了奇异值信息量的90%:", n90
		n90 = lsa.maxWeight(s, 0.5)
		print "前项目占据了奇异值信息量的90%:", n90
		n90 = lsa.maxWeight(s, 0.6)
		print "前项目占据了奇异值信息量的90%:", n90
