# _*_ coding: utf-8 _*_
from hmgis.Test.SVDTest import *


class Recommend:
	## 欧式距离
	def ecludSim(self, inA, inB):
		return 1.0 / (1.0 + la.norm(inA - inB))

	## 皮尔逊距离
	def pearsSim(self, inA, inB):
		if len(inA) < 3: return 1.0
		return 0.5 + 0.5 * corrcoef(inA, inB, rowvar=0)[0][1]

	## COS距离
	def cosSim(self, inA, inB):
		num = float(inA.T * inB)
		denom = la.norm(inA) * la.norm(inB)
		return 0.5 + 0.5 * (num / denom)

	def standEst(self, dataMat, user, simMeas, item):
		n = shape(dataMat)[1]
		simTotal = 0.0;
		ratSimTotal = 0.0
		for j in range(n):
			userRating = dataMat[user, j]
			if userRating == 0: continue
			overLap = nonzero(logical_and(dataMat[:, item].A > 0, \
			                              dataMat[:, j].A > 0))[0]
			if len(overLap) == 0:
				similarity = 0
			else:
				similarity = simMeas(dataMat[overLap, item], \
				                     dataMat[overLap, j])
			print 'the %d and %d similarity is: %f' % (item, j, similarity)
			simTotal += similarity
			ratSimTotal += similarity * userRating
		if simTotal == 0:
			return 0
		else:
			return ratSimTotal / simTotal

	def svdEst(self, dataMat, user, simMeas, item):
		n = shape(dataMat)[1]
		simTotal = 0.0;
		ratSimTotal = 0.0
		U, Sigma, VT = la.svd(dataMat)
		Sig4 = mat(eye(4) * Sigma[:4]) #arrange Sig4 into a diagonal matrix
		xformedItems = dataMat.T * U[:, :4] * Sig4.I  #create transformed items
		print xformedItems
		for j in range(n):
			userRating = dataMat[user, j]
			if userRating == 0 or j == item: continue
			similarity = simMeas(xformedItems[item, :].T, \
			                     xformedItems[j, :].T)
			print 'the %d and %d similarity is: %f' % (item, j, similarity)
			simTotal += similarity
			ratSimTotal += similarity * userRating
		if simTotal == 0:
			return 0
		else:
			return ratSimTotal / simTotal

	def recommend(self, dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
		## 找出用户对哪些物品没有进行过打分
		unratedItems = nonzero(dataMat[user, :].A == 0)[1]#find unrated items
		if len(unratedItems) == 0: return 'you rated everything'
		itemScores = []
		for item in unratedItems:
			estimatedScore = estMethod(dataMat, user, simMeas, item)
			itemScores.append((item, estimatedScore))
		return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N]

	def printMat(self, inMat, thresh=0.8):
		for i in range(32):
			for k in range(32):
				if float(inMat[i, k]) > thresh:
					print 1,
				else:
					print 0,
			print ''


