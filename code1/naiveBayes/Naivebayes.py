import numpy
import os
import scipy.io as sio
from scipy.stats import norm


def main():
    # laod data
    mat = sio.loadmat("data")
    trainX = mat['Xtrn']
    trainY = mat['Ytrn']
    trainY = trainY[:, 0]
    testX = mat['Xtst']
    testY = mat['Ytst']
    testY = testY[:, 0]

    m, n = trainX.shape
    index0 = numpy.where(trainY == 0)[0]
    #index0 = list(index0)
    index1 = numpy.where(trainY == 1)[0]
    #index1 = list(index1)

    # separate data by trainY = 0 or 1
    trainX0 = trainX[index0, :]
    trainX1 = trainX[index1, :]

    # generate normal distribution for each feature
    trainX0_mean = numpy.mean(trainX0, axis=0)
    trainX1_mean = numpy.mean(trainX1, axis=0)
    trainX0_std = numpy.std(trainX0, axis=0)
    trainX1_std = numpy.std(trainX1, axis=0)

    list0 = []
    list1 = []
    for num in range(0, n):
        list0.append(norm(trainX0_mean[num], trainX0_std[num]))
        list1.append(norm(trainX1_mean[num], trainX1_std[num]))

    # build prob matrix for 0 and 1
    train_prob0 = numpy.array([0.0] * (m * n)).reshape(m, n)
    train_prob1 = numpy.array([0.0] * (m * n)).reshape(m, n)

    for num in range(0, n):
        train_prob0[:, num] = numpy.log(list0[num].pdf(trainX[:, num]))
        train_prob1[:, num] = numpy.log(list1[num].pdf(trainX[:, num]))

    # column sum
    prob0 = numpy.sum(train_prob0, axis=1)
    prob1 = numpy.sum(train_prob1, axis=1)

    pred = numpy.array([0] * m)
    numpy.putmask(pred, prob1 > prob0, 1)

    trainErr = float(sum(trainY != pred)) / m

    print "train err=", trainErr
    print "trainX=", trainX.shape
    # print "trainY=", trainY.shape


if __name__ == '__main__':
    main()
