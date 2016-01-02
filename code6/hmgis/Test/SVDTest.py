# _*_ coding: utf-8 _*_

from numpy import *
from numpy import linalg as la

## SVD测试计算
class SVDTest:
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

	## SVD第一个测试，可以看出SIGMA的值有5项，但前2项远大于后3项
	## 这样实际上原始矩阵Data = Um2*Sigma22*VT2n来表示
	def svdTest1(self):
		data = self.loadExData()
		## 使用numpy的SVD技术分解矩阵
		U, Sigma, VT = la.svd(data)
		print U
		print '--sigma-----------------------------'
		print Sigma
		print '--vt--------------------------------'
		print VT
		## 将Sigma的主要值提取出来
		## 构造一个对角矩阵
		print '--------------------------------------'
		Sig3 = mat([[Sigma[0], 0], [0, Sigma[1]]])
		## 下面这个矩阵和原矩阵非常类似
		print data
		print U[:, :2] * Sig3 * VT[:2, :]

	## 计算缩减的sigma，可以获得表示90%量的
	def svdTest2(self):
		data = self.loadExData2()
		print mat(data).A
		U, Sigma, VT = la.svd(data)
		print mat(U).A
		print mat(Sigma).A
		print mat(VT).A
		sig2 = Sigma ** 2
		print "--------------------"
		print "sigma值总量", sum(sig2)
		print "sigma值90%", sum(sig2) * 0.9
		##
		print sum(sig2[:2])
		print sum(sig2[:3])
