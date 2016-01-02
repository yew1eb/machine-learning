import csv
import random
import numpy
import LogisticRegression as LR

#import data
data = numpy.genfromtxt('input.csv', delimiter=',')
# response is in the first column
Y = data[:, 0]
X = data[:, 1:]

# n-fold cross validation
# shuffle data
m = len(Y)
index = range(0, m)
random.shuffle(index)
X = X[index, :]
Y = Y[index]

# n-fold
nfold = 10
foldSize = int(m / nfold)

# arrage to store training and testing error
trainErr = [0.0] * nfold
testErr = [0.0] * nfold
allIndex = range(0, m)
for i in range(0, nfold):

    testIndex = range((foldSize * i), foldSize * (i + 1))
    trainIndex = list(set(allIndex) - set(testIndex))

    trainX = X[trainIndex, :]
    trainY = Y[trainIndex]
    testX = X[testIndex, :]
    testY = Y[testIndex]

    # set parameter
    alpha = 0.05
    lam = 0.1
    model = LR.LogisticRegression(trainX, trainY, alpha, lam)
    model.run(400, printIter=False)

    trainPred = model.predict(trainX)
    trainErr[i] = float(sum(trainPred != trainY)) / len(trainY)

    testPred = model.predict(testX)
    testErr[i] = float(sum(testPred != testY)) / len(testY)

    print "train Err=", trainErr[i], "test Err=", testErr[i]
    print " "

print "summary:"
print "average train err =", numpy.mean(trainErr) * 100, "%"
print "average test err =", numpy.mean(testErr) * 100, "%"
