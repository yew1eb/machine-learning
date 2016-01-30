# -*- coding: utf-8 -*-
'''    
    @Description : dualperceptron by python
    @Author: Liu_Longpo
    @Time: Sun Dec 20 12:57:00 2015
'''

import matplotlib.pyplot as plt
import numpy as np
import time
 
trainSet = []
 
w = []
a = []
b = 0
lens = 0
alpha = 0
Gram = []
trainLoss = []
 
def calInnerProduct(i, j):
    global lens
    res = 0
    for p in range(lens):
        res += trainSet[i][0][p] * trainSet[j][0][p]
    return res

def AddVector(vec1, vec2):
    retvec = []
    for i in range(len(vec1)):
        retvec.append(vec1[i] + vec2[i])
    return retvec
    
def NumProduct(num, vec):
    retvec = []
    for i in range(len(vec)):
        retvec.append(num * vec[i])
    return retvec 

def createGram():
    global lens
    for i in range(len(trainSet)):
        tmp = []
        for j in range(0, len(trainSet)):
            tmp.append(calInnerProduct(i, j))
        Gram.append(tmp)

# update parameters using stochastic gradient descent
def updateParm(k):
    global a, b, alpha
    a[k] += alpha
    b = b + alpha * trainSet[k][1] 
    #print a, b # you can uncomment this line to check the process of stochastic gradient descent
 
def calDistance(k):
    global a, b
    res = 0
    for i in range(len(trainSet)):
        res += a[i] * int(trainSet[i][1]) * Gram[i][k]
    res += b
    res *= trainSet[k][1]
    return res
 
def trainModel(Iter):
    print "training MLP..."
    print "-"*40
    epoch = 0
    for i in range(Iter):
        train_loss = 0
        global w, a
        update = False
        print "epoch",epoch, "  w: ",w,"b:",b,
        for j in range(len(trainSet)):
            res = calDistance(j)
            if res <= 0:
                train_loss += -res
                update = True
                updateParm(j)
        print 'train loss:',train_loss
        trainLoss.append(train_loss)
        if update:
            epoch = epoch+1
        else:
            for k in range(len(trainSet)):
                w = AddVector(w, NumProduct(a[k] * int(trainSet[k][1]), trainSet[k][0]))
            print "result: w: ", w, " b: ", b
        update = False
        if epoch==Iter:
            print 'reach max trian epoch'
            for j in range(len(trainSet)):
                w = AddVector(w, NumProduct(a[j] * int(trainSet[j][1]), trainSet[j][0]))
            print "RESULT: w: ", w, " b: ", b
 
if __name__=="__main__":
    if len(sys.argv)!=4:
        print "Usage: python MLP.py trainFile modelFile"
        exit(0)
    alpha = float(sys.argv[1])
    trainFile = open(sys.argv[2])
    modelPath = sys.argv[3]
    #modelPath = 'model'
    lens = 0
    # load data  trainSet[i][0]:data,trainSet[i][1]:label
    for line in trainFile:
        data = line.strip().split('\t')
        lens = len(data) - 1
        sample_all = []
        sample_data = []
        for i in range(0,lens):
            sample_data.append(float(data[i]))
        sample_all.append(sample_data) # add data
        if int(data[lens]) == 1:
            sample_all.append(int(data[lens])) # add label
        else:
            sample_all.append(-1) # add label
        trainSet.append(sample_all)
    trainFile.close()
    createGram()
    for i in range(len(trainSet)):
        a.append(0)
    for i in range(lens):
        w.append(0)
    start = time.clock()
    trainModel(500)
    end = time.clock()
    print 'train time is %f s' % (end - start)
    x = np.linspace(-5,5,10)
    plt.figure()
    for i in range(len(trainSet)):
        if trainSet[i][1] == 1:
            plt.scatter(trainSet[i][0][0],trainSet[i][0][1],c=u'b')
        else:
            plt.scatter(trainSet[i][0][0],trainSet[i][0][1],c=u'r')
    plt.plot(x,-(w[0]*x+b)/w[1],c=u'r')
    plt.show()
    trainIter = range(len(trainLoss))
    plt.figure()    
    plt.scatter(trainIter,trainLoss,c=u'r')
    plt.plot(trainIter,trainLoss)
    plt.xlabel('Epoch')
    plt.ylabel('trainLoss')
    plt.show()
    
