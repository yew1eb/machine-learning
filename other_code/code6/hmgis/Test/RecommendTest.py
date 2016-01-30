# _*_ coding: utf-8 _*_
from hmgis.TextMining.Recommend import *


class RecommendTest:
	## 加载简单数据
	def loadExData(self):
		return [[0, 0, 0, 2, 2],
		        [0, 0, 0, 3, 3],
		        [0, 0, 0, 1, 1],
		        [1, 1, 1, 0, 0],
		        [2, 2, 2, 0, 0],
		        [5, 5, 5, 0, 0],
		        [1, 1, 1, 0, 0]]

	## 加载一个稀疏矩阵数据
	def loadExData2(self):
		return [[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
		        [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
		        [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
		        [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
		        [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
		        [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
		        [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
		        [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
		        [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
		        [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
		        [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]

	## 基于物品相似度的推荐
	def recommendTest(self):
		myMat = mat(self.loadExData())
		print myMat
		myMat[0, 1] = myMat[0, 0] = myMat[1, 0] = myMat[2, 0] = 4
		myMat[3, 3] = 2
		print myMat
		## 第一个参数为数据，第二个参数为用户，第三个参数为数量
		## 结果为[(2, 2.5), (1, 2.0243290220056256)]，表示用户对第1和2项的估计评分
		recommend = Recommend()
		print recommend.recommend(myMat, 2, 5, recommend.cosSim, recommend.standEst)

	def recommendTest2(self):
		myMat = mat(self.loadExData2())
		## 第一个参数为数据，第二个参数为用户，第三个参数为数量
		## 结果为[(2, 2.5), (1, 2.0243290220056256)]，表示用户对第1和2项的估计评分
		recommend = Recommend()
		print recommend.recommend(myMat, 2, 8, recommend.cosSim, recommend.svdEst)
		print recommend.recommend(myMat, 2, 8, recommend.cosSim, recommend.standEst)

	def singleInfoSimilary(self):
		myMat = mat(self.loadExData())
		## 以下为矩阵行数据的相似度测试
		recom = Recommend()
		print recom.ecludSim(myMat[:, 0], myMat[:, 4])
		print recom.cosSim(myMat[:, 0], myMat[:, 4])
		print recom.cosSim(myMat[0, :].T, myMat[4, :].T)
		print recom.pearsSim(myMat[:, 0], myMat[:, 4])

