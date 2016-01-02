from __future__ import division
import csv, math, random
from collections import defaultdict
import pandas as pd
from sklearn.naive_bayes import GaussianNB

def loadCsv(filename):
	lines = csv.reader(open(filename, 'rb'))
	return [[float(y) for y in x] for x in list(lines)]

# filename = 'pima-indians-diabetes.data.csv'
# dataset = loadCsv(filename)
# print('Loaded data file {0} with {1} rows').format(filename, len(dataset))

def splitDataset(dataset, splitRatio):
	copy = list(dataset)
	random.shuffle(copy)
	dividePoint = int(len(copy) * splitRatio)
	return (copy[:dividePoint], copy[dividePoint:])

# dataset = [[1], [2], [3], [4], [5]]
# splitRatio = 0.67
# train, test = splitDataset(dataset, splitRatio)
# print('Split {0} rows into train with {1} and test with {2}').format(len(dataset), train, test)

def separateByClass(dataset):
	separated = {}
	for vector in dataset:
		if (vector[-1] not in separated):
			separated[vector[-1]] = []
		separated[vector[-1]].append(vector)
	return separated

# dataset = [[1,20,1], [2,21,0], [3,22,1]]
# separated = separateByClass(dataset)
# print('Separated instances: {0}').format(separated)

def mean(numbers):
	return sum(numbers)/len(numbers)

def stdev(numbers):
	avg = mean(numbers)
	variance = sum([(x-avg)**2 for x in numbers])/(len(numbers)-1)
	return math.sqrt(variance)

# numbers = [1,2,3,4,5]
# print('Summary of {0}: mean={1}, stdev={2}').format(numbers, mean(numbers), stdev(numbers))

def summarize(dataset):
	summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
	return summaries[:-1]

# dataset = [[1,20,0], [2,21,1], [3,22,0]]
# summary = summarize(dataset)
# print('Attribute summaries: {0}').format(summary)

def summarizeByClass(dataset):
	separated = separateByClass(dataset)
	return {classValue: summarize(instances) for classValue, instances in separated.iteritems()}

# dataset = [[1,20,1], [2,21,0], [3,22,1], [4,22,0]]
# summary = summarizeByClass(dataset)
# print('Summary by class value: {0}').format(summary)

def calculateProbability(x, mean, stdev):
	"""Gaussian distribution"""
	exponent = math.exp(- ((x - mean)**2 / (2 * stdev**2)))
	return exponent / (math.sqrt(2 * math.pi) * stdev)

# x = 71.5; mean = 73; stdev = 6.2
# probability = calculateProbability(x, mean, stdev)
# print('Probability of belonging to this class: {0}').format(probability)

def calculateClassProbabilities(summaries, inputVector):
	probabilities = {}
	for classValue, classSummaries in summaries.iteritems():
		probabilities[classValue] = 0
		for s, x in zip(classSummaries, inputVector):
			mean, stdev = s
			probabilities[classValue] += math.log(calculateProbability(x, mean, stdev))
	return probabilities

# summaries = {0:[(1, 0.5)], 1:[(20, 5.0)]}
# inputVector = [1.1, '?']
# probabilities = calculateClassProbabilities(summaries, inputVector)
# print('Probabilities for each class: {0}').format(probabilities)

def predict(summaries, inputVector):
	probabilities = calculateClassProbabilities(summaries, inputVector)
	return max(probabilities, key=probabilities.get)

# summaries = {'A':[(1, 0.5)], 'B':[(20, 5.0)]}
# inputVector = [1.1, '?']
# result = predict(summaries, inputVector)
# print('Prediction: {0}').format(result)

def getPredictions(summaries, testSet):
	return [predict(summaries, i) for i in testSet]

# summaries = {'A':[(1, 0.5)], 'B':[(20, 5.0)]}
# testSet = [[1.1, '?'], [19.1, '?']]
# predictions = getPredictions(summaries, testSet)
# print('Predictions: {0}').format(predictions)

def getAccuracy(testSet, predictions):
	return len([1 for t, p in zip(testSet, predictions) if t[-1] == p]) / len(testSet)

# testSet = [[1,1,1,'a'], [2,2,2,'a'], [3,3,3,'b']]
# predictions = ['a', 'a', 'a']
# accuracy = getAccuracy(testSet, predictions)
# print('Accuracy: {0}').format(accuracy)

def main():
	filename = 'pima-indians-diabetes.data.csv'
	splitRatio = 0.67
	dataset = pd.read_csv(filename)
	# dataset = loadCsv(filename)
	trainingSet, testSet = splitDataset(dataset, splitRatio)
	# prepare model
	summaries = summarizeByClass(trainingSet)
	# test model
	predictions = getPredictions(summaries, testSet)
	accuracy = getAccuracy(testSet, predictions)
	print('Accuracy: {0}%').format(accuracy)
 
# if __name__=='__main__': main()



from sklearn import datasets

iris = datasets.load_iris()
gnb = GaussianNB()

# print iris.data.shape
# print iris.target.shape

# print iris.data[:2]
# print iris.target[:2]

y_fit = gnb.fit(iris.data, iris.target)
y_pred = y_fit.predict(iris.data)

print y_pred

# print("mislabeled points out of a total %d points : %d" % (iris.data.shape[0],(iris.target != y_pred).sum()))
