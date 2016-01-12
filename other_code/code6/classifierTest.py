# _*_ coding: utf-8 _*_
#
# author : chiangbt@gmail.com
# function : 整个包的测试起点
# 涵盖：
# 分类包：KNN、DecisionTree、Bayesian、Logistic、SVM_R、AdaBoostTest
# 回归包：线性回归、多元线性回归
# 聚类包：K-Means
# 文本挖掘：SVD、LSI、pLSA、LDA
# 需要安装的第三方包：numpy、scipy、nltk、scikit、gensim
#

import sys
from hmgis.Test.KNNTest import *
from hmgis.Test.DecisionTreeTest import *
from hmgis.Test.BayesianTest import *
from hmgis.Test.LogisticTest import *
from hmgis.Test.SVMTest import *
from hmgis.Test.AdaBoostTest import *
from hmgis.Test.RegressionTest import *
from hmgis.Test.KMeansTest import *
from hmgis.Test.SVDTest import *
from hmgis.Test.RecommendTest import *
from hmgis.Test.LSATest import *
from hmgis.Test.GensimTest.SimpledocTest import *
from hmgis.Test.pLSATest import *


class ClassifierTest:
	## KNN距离分类测试，KNN分类适合数值型分类
	def knnTest(self):
		knn = KNNTest()
		## 最简单的KNN测试，判断一个点的分类情况
		knn.knnTest()
		## 点的可视化
		knn.show2DPoint('data/knn1/datingTestSet2.txt')
		knn.show3DPoint('data/knn1/datingTestSet2.txt')
		## 从txt文件中获得数据进行分类
		## 分类前会进行归一化处理
		knn.knnTest2('data/knn1/datingTestSet2.txt')
		print '-----------------------'
		## 使用scikit learn包中的KNN函数进行分类
		knn.knnTestScikit('data/knn1/datingTestSet2.txt')

	## 决策树分类
	def dtTest(self):
		dt = DecisionTreeDemo()
		dataMat, labels = dt.createDataset('data/dt/lenses.txt')
		## 由于决策树分类的可视化结果比较麻烦，直接使用了scikit的方法
		dt.dtTest(dataMat, labels)

	## 贝叶斯分类，适用于文本分类情况
	def bayesianTest(self):
		bayesian = BayesianTest()
		listOPosts, listClasses = bayesian.loadDataSet()
		# ## 最简单的贝叶斯分类
		# bayesian.testingNB()
		# ## 从RSS文件中进行分类
		rssBayesian = RSSBayesianTest()
		# rssBayesian.SingleClassifier()
		# rssBayesian.scikitNBClassfier()
		# rssBayesian.crossValidClassifier()
		# ## 对email数据进行分类
		emailBayesian = emailClassfier()
		emailBayesian.spamTest(bayesian)

	## Logistic分类，将数据分为0和1两种类型
	def logisticTest(self):
		logic = LogisticTest()
		## t梯度下降法计算
		logic.calGradAscent('data/logistic/testSet.txt')
		## 随机梯度下降法
		logic.calRandomGradAscent('data/logistic/testSet.txt')
		## 改进随机梯度下降法
		logic.calRandomGradAscent2('data/logistic/testSet.txt')
		## 交叉验证
		logic.multiColicHorseTest()
		## SCIKIT的logistic分类
		logic.calScikitLogistic('data/logistic/testSet.txt')

	## SVM分类，原理太复杂
	def svmTest(self):
		svm = SVMTest()
		## 线性SVM
		# svm.testLinear()
		# svm.testMultiLinear()
		# ## RBF核函数
		# svm.testRbf(('rbf', 1.5))
		svm.testDigits(('rbf', 50))

	# ## SciKit中的SVM方法
	# svm.testSciKitSVM()

	## Boost类型分类，它将一个简单的弱分类器单根决策树构成了一个强分类器
	def adaBoostTest(self):
		ada = AdaBoostTest()
		ada.adaboostTest()

	#ada.scikitAdaboost()

	## 回归分析
	def regressionTest(self):
		reg = RegressionTest()
		reg.simpleTest()
		reg.multiTest()
		#
		#岭回归
		reg.ridgeRegression()

	## KMeans聚类
	def kmeansTest(self):
		kmeans = KMeansTest()
		kmeans.kMeansTest()
		kmeans.KMeansTest2()
		kmeans.KMeansTest3()
		kmeans.ClusterClubsTest(6)

	#kmeans.ScikitKMeansTest()

	## SVD降维分解，它是LSI的核心
	def SVDTest(self):
		svd = SVDTest()
		svd.svdTest1()
		svd.svdTest2()

	## 推荐系统，主要是相似度计算
	def recommendTest(self):
		recom = RecommendTest()
		recom.recommendTest()
		recom.recommendTest2()
		recom.singleInfoSimilary()

	## LSA（LSI）测试
	def LSATest(self):
		lsa = LSATest()
		# lsa.simpleTest()
		# lsa.englishCorpusTest()
		lsa.weiboTest()

	## Gensim库的测试
	def GensimTest(self):
		gen = GensimTest()
		# gen.simple()
		# gen.simple2()
		gen.GIS3SNewsTopic()

	# gen.weiboTopic()

	def pLSATest(self):
		plsa = pLSATest()
		plsa.plsaTest()


if __name__ == "__main__":
	reload(sys)                         # 2
	sys.setdefaultencoding('utf-8')
	sys.getfilesystemencoding()

	classifier = ClassifierTest()
	# classifier.knnTest()
	classifier.dtTest()
# classifier.bayesianTest()
# classifier.logisticTest()
# classifier.svmTest()
# classifier.adaBoostTest()
# classifier.regressionTest()
#classifier.kmeansTest()
#classifier.SVDTest()
# classifier.recommendTest()
# classifier.LSATest()
# classifier.GensimTest()
# classifier.pLSATest()
