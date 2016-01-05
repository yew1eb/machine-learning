# _*_ coding: utf-8 _*_
from numpy import *


class Kmeans:
	## 欧几里得距离
	def distEclud(self, vecA, vecB):
		return sqrt(sum(power(vecA - vecB, 2))) #la.norm(vecA-vecB)

	## 球面距离
	def distSLC(self, vecA, vecB):#Spherical Law of Cosines
		a = sin(vecA[0, 1] * pi / 180) * sin(vecB[0, 1] * pi / 180)
		b = cos(vecA[0, 1] * pi / 180) * cos(vecB[0, 1] * pi / 180) * \
		    cos(pi * (vecB[0, 0] - vecA[0, 0]) / 180)
		return arccos(a + b) * 6371.0 #pi is imported with numpy

	## k为随机质心的数量
	def randCent(self, dataSet, k):
		n = shape(dataSet)[1]
		centroids = mat(zeros((k, n)))#create centroid mat
		dataSet = asarray(dataSet)
		for j in range(n):#create random cluster centers, within bounds of each dimension
			minJ = min(dataSet[:, j])
			rangeJ = float(max(dataSet[:, j]) - minJ)
			centroids[:, j] = mat(minJ + rangeJ * random.rand(k, 1))
		return centroids

	##-------------------------------------------------------------------------
	## 一般KMeans方法
	def kMeans(self, dataSet, k, distMeas=distEclud, createCent=randCent):
		m = shape(dataSet)[0]
		clusterAssment = mat(zeros((m, 2)))#create mat to assign data points
		#to a centroid, also holds SE of each point
		centroids = createCent(dataSet, k)
		clusterChanged = True
		while clusterChanged:
			clusterChanged = False
			for i in range(m):#for each data point assign it to the closest centroid
				minDist = inf;
				minIndex = -1
				for j in range(k):
					distJI = distMeas(centroids[j, :], dataSet[i, :])
					if distJI < minDist:
						minDist = distJI;
						minIndex = j
				if clusterAssment[i, 0] != minIndex: clusterChanged = True
				clusterAssment[i, :] = minIndex, minDist ** 2
			print centroids
			for cent in range(k):#recalculate centroids
				ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]#get all the point in this cluster
				centroids[cent, :] = mean(ptsInClust, axis=0) #assign centroid to mean
		return centroids, clusterAssment

	# 改进后的Kmeans方法，即二分K均值法
	def biKmeans(self, dataSet, k, distMeas=distEclud, createCent=randCent):
		m = shape(dataSet)[0]
		clusterAssment = mat(zeros((m, 2)))
		centroid0 = mean(dataSet, axis=0).tolist()[0]
		centList = [centroid0] #create a list with one centroid
		for j in range(m):#calc initial Error
			clusterAssment[j, 1] = distMeas(mat(centroid0), dataSet[j, :]) ** 2
		while (len(centList) < k):
			lowestSSE = inf
			for i in range(len(centList)):
				ptsInCurrCluster = dataSet[nonzero(clusterAssment[:, 0].A == i)[0],
				                   :]#get the data points currently in cluster i
				centroidMat, splitClustAss = self.kMeans(ptsInCurrCluster, 2, distMeas, createCent)
				sseSplit = sum(splitClustAss[:, 1])#compare the SSE to the currrent minimum
				sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:, 0].A != i)[0], 1])
				print "sseSplit, and notSplit: ", sseSplit, sseNotSplit
				if (sseSplit + sseNotSplit) < lowestSSE:
					bestCentToSplit = i
					bestNewCents = centroidMat
					bestClustAss = splitClustAss.copy()
					lowestSSE = sseSplit + sseNotSplit
			bestClustAss[nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList) #change 1 to 3,4, or whatever
			bestClustAss[nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
			print 'the bestCentToSplit is: ', bestCentToSplit
			print 'the len of bestClustAss is: ', len(bestClustAss)
			centList[bestCentToSplit] = bestNewCents[0, :].tolist()[0]#replace a centroid with two best centroids
			centList.append(bestNewCents[1, :].tolist()[0])
			clusterAssment[nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0],
			:] = bestClustAss#reassign new clusters, and SSE
		return mat(centList), clusterAssment
