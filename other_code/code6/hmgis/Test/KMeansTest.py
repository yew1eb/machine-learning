# _*_ coding: utf-8 _*_
from numpy import *
import matplotlib.pyplot as plt
from hmgis.Clustering.KMean import *


class KMeansTest:
	def loadDataSet(self, fileName):      #general function to parse tab -delimited floats
		dataMat = []                #assume last column is target value
		fr = open(fileName)
		for line in fr.readlines():
			curLine = line.strip().split('\t')
			fltLine = map(float, curLine) #map all elements to float()
			dataMat.append(fltLine)
		return dataMat

	## 计算KMeans聚类的准备工作
	def kMeansTest(self):
		dataMat = self.loadDataSet('data/kmean/testSet.txt')
		## 将List转换为ndarray
		dataMat = asarray(dataMat)

		kmeans = Kmeans()
		# 计算两个质心点坐标
		print kmeans.randCent(dataMat, 2)
		## 两个质点间的距离
		print kmeans.distEclud(dataMat[0], dataMat[1])

	##以下为测试方法
	## 一般KMEANS聚类
	def KMeansTest2(self):
		dataMat = mat(self.loadDataSet('data/kmean/testSet.txt'))

		kmeans = Kmeans()
		myCentroids, clustAssing = kmeans.kMeans(dataMat, 4, kmeans.distEclud, kmeans.randCent)
		## 质心点
		print myCentroids
		## 每个原始点的分类
		print clustAssing

	## 二分KMENAS聚类
	def KMeansTest3(self):
		dataMat = mat(self.loadDataSet('data/kmean/testSet2.txt'))

		kmeans = Kmeans()
		centList, newAssment = kmeans.biKmeans(dataMat, 3, kmeans.distEclud, kmeans.randCent)
		print centList

	def ClusterClubsTest(self, numClust=5):
		datList = []
		for line in open('data/kmean/places.txt').readlines():
			lineArr = line.split('\t')
			datList.append([float(lineArr[4]), float(lineArr[3])])
		datMat = mat(datList)

		kmeans = Kmeans()
		myCentroids, clustAssing = kmeans.biKmeans(datMat, numClust, kmeans.distSLC, kmeans.randCent)
		fig = plt.figure()
		rect = [0.1, 0.1, 0.8, 0.8]
		scatterMarkers = ['s', 'o', '^', '8', 'p', \
		                  'd', 'v', 'h', '>', '<']
		axprops = dict(xticks=[], yticks=[])
		ax0 = fig.add_axes(rect, label='ax0', **axprops)
		imgP = plt.imread('data/kmean/Portland.png')
		ax0.imshow(imgP)
		ax1 = fig.add_axes(rect, label='ax1', frameon=False)
		for i in range(numClust):
			ptsInCurrCluster = datMat[nonzero(clustAssing[:, 0].A == i)[0], :]
			markerStyle = scatterMarkers[i % len(scatterMarkers)]
			ax1.scatter(ptsInCurrCluster[:, 0].flatten().A[0], ptsInCurrCluster[:, 1].flatten().A[0],
			            marker=markerStyle, s=90)
		ax1.scatter(myCentroids[:, 0].flatten().A[0], myCentroids[:, 1].flatten().A[0], marker='+', s=300)
		plt.show()


	def ScikitKMeansTest(self):
		dataMat = self.loadDataSet('data/kmean/testSet.txt')
		from sklearn.cluster import KMeans

		kmeans = KMeans(init='k-means++', n_clusters=5, n_init=10)
		kmeans.fit(dataMat)

		import numpy as np
		import pylab as pl

		h = .02     # point in the mesh [x_min, m_max]x[y_min, y_max].
		reduced_data = np.array(dataMat)
		# Plot the decision boundary. For that, we will assign a color to each
		x_min, x_max = reduced_data[:, 0].min() + 1, reduced_data[:, 0].max() - 1
		y_min, y_max = reduced_data[:, 1].min() + 1, reduced_data[:, 1].max() - 1
		xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

		# Obtain labels for each point in mesh. Use last trained model.
		Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
		# Put the result into a color plot
		Z = Z.reshape(xx.shape)

		pl.imshow(Z, interpolation='nearest',
		          extent=(xx.min(), xx.max(), yy.min(), yy.max()),
		          cmap=pl.cm.Paired,
		          aspect='auto', origin='lower')

		pl.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
		# Plot the centroids as a white X
		centroids = kmeans.cluster_centers_
		print centroids
		pl.scatter(centroids[:, 0], centroids[:, 1],
		           marker='x', s=169, linewidths=3,
		           color='w', zorder=10)
		pl.xlim(x_min, x_max)
		pl.ylim(y_min, y_max)

		pl.show()