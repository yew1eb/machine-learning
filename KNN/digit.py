#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Filename: digit.py
@Author: yew1eb
@Date: 2015/12/21 0021
"""

## 手写识别系统
#这里可以将手写字符看做由01组成的32*32个二进制文件，然后转换为1*1024的向量即为一个训练样本，每一维即为一个特征值
### 1 将一个32*32的二进制图像转换成1*1024的向量
def img2vector(filename):
    returnVect = np.zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect

### 2 手写识别系统测试代码

#手写识别系统测试代码
def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')   #获取目录内容
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]              #分割得到标签  从文件名解析得到分类数据
        fileStr = fileNameStr.split('.')[0]
        classStr = int(fileStr.split('_')[0])
        hwLabels.append(classStr)                 #测试样例标签
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print 'the classifier came back with: %d, the real answer is: %d' % (classifierResult, classStr)
        if(classifierResult != classStr): errorCount += 1.0
    print "\nthe total numbers of errors is : %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount/float(mTest))

