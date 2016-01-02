# from sklearn.naive_bayes import GaussianNB
from sklearn import datasets

import csv, math, random
import pandas as pd
import numpy as np

class GaussianNB(object):

	def __init__(self):
		self.model = None

	def fit(self, X, y):
		"""X : array-like, shape (n_samples, n_features)
			y : array-like, shape (n_samples,)"""
		# scheme of separated
		# {0: [[2, 21]], 1: [[1, 20], [3, 22]]}
		separated = {c: [x for x, t in zip(X, y) if t==c] for c in np.unique(y)}
		self.model = {c: [(np.mean(attr), np.std(attr)) for attr in zip(*instances)]
						for c, instances in separated.items()}
		return self

	def prob(self, x, mean, std):
		"""Gaussian distribution in log"""
		exponent = math.exp(- ((x - mean)**2 / (2 * std**2)))
		return math.log(exponent / (math.sqrt(2 * math.pi) * std))

	def predictOne(self, X):
		probs = {c: sum( self.prob(x, *s) for s, x in zip(summaries, X) )
				for c, summaries in self.model.items()}
		return max(probs, key=probs.get)

	def predict(self, X):
		"""	X : array-like, shape = [n_samples, n_features]
			return : array, shape = [n_samples]"""
		return [self.predictOne(i) for i in X]



def main():
	iris = datasets.load_iris()
	gnb = GaussianNB()
	y_fit = gnb.fit(iris.data, iris.target)
	y_pred = y_fit.predict(iris.data)
	print("out of a total %d points : %d" % (iris.data.shape[0],(iris.target != y_pred).sum()))
 
if __name__=='__main__': main()


# X = np.array([[1, 20], [2, 21], [3, 22], [4,22]])
# y = np.array([1, 0, 1, 0])
# gnm = GaussianNB()
# gnm.fit(X, y)
# print(gnm.predict(np.array([[1.1], [19.1]])))
