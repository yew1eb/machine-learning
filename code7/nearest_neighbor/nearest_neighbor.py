from sklearn import datasets

import csv, math, random
import pandas as pd
import numpy as np
from collections import defaultdict




class KNeighborsClassifier(object):

	def __init__(self, n_neighbors=5, weights='uniform'):
		self.n_neighbors = n_neighbors
		self.weights = weights

	def fit(self, X, y):
		self.X = X
		self.y = y
		return self

	def distance(self, data1, data2):
		"""manhattan distance"""
		return sum(abs(i) for i in data1 - data2)

	def compute_weights(self, distances):
		if self.weights == 'uniform':
			return [(1, y) for d, y in distances]
		else: # 'distance'
			return [(1/d, y) for d, y in distances]

	def predictOne(self, test):
		distances = sorted((self.distance(x, test), y) for x, y in zip(self.X, self.y))
		weights = self.compute_weights(distances[:self.n_neighbors])
		weights_by_class = defaultdict(list)
		for d, c in weights:
			weights_by_class[c].append(d)
		return max((sum(val), key) for key, val in weights_by_class.items())[1]

	def predict(self, X):
		"""	X : array-like, shape = [n_samples, n_features]
			return : array, shape = [n_samples]"""
		return [self.predictOne(i) for i in X]


def main():
	iris = datasets.load_iris()
	neig = KNeighborsClassifier()
	y_fit = neig.fit(iris.data, iris.target)
	y_pred = y_fit.predict(iris.data)	
	print("out of a total %d points : %d" % (iris.data.shape[0],(iris.target != y_pred).sum()))

	# X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
	# y = np.array([1,1,1,0,0,0])

	# neig = KNeighborsClassifier(weights='distance').fit(X, y)
	# print(neig.predictOne(np.array([0,0])))


if __name__ == '__main__': main()








