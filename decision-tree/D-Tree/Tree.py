from math import log
# 计算熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannoEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannoEnt -= prob * log(prob,2)
    return shannoEnt
def createDataSet():
    dataSet = [[1,1,0,'yes'],[1,1,1,'yes'],[1,0,1,'no'],[0,1,0,'no'],[0,1,0,'no']]
    label = ['no surfacing','flippers']
    return dataSet,label
# 划分数据集
# axis为特征，也就是对某一个特征进行判定，比如身高
# value为特征的值，也就是说，某一个特征的不同值，
# 比如身高这个特征有高和矮之分，高跟矮就是这个value
# 数据集为 [ [1,1,0,'yes'],[1,1,1,'yes'],[1,0,1,'no'],[0,1,0,'no'],[0,1,0,'no'] ]
# 每一个 feaVec 就是一行，也就是一个样本数据
def spiltDataSet(dataSet,axis,value):
    retDataSet = []
    for featVec in dataSet:
        # 如果当前样本数据的特征 axis 的值 featVec[axis] 与我们要求的value相等
        # 也就是我们认为当前的样本符合我们的要求
        if featVec[axis] == value:
        # 对于符合要求的样本数据
        # 我们将这个特征剪掉，剩下的数据组成一个新的子样本并返回
            redecedFeatVec = featVec[:axis]
            redecedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(redecedFeatVec)
    return retDataSet
# 寻找最优的划分特征
def chooseBestFeatureToSplit(dataSet):
    # 为什么减一？
    numFeatures = len(dataSet[0])-1
    # 计算源数据的熵
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in xrange(numFeatures):
        # 获取每一列，因为每一列都属于一个特征的不同value
        featList = [example[i] for example in dataSet]
        # 确保每个值都是唯一的
        uniqueVals = set(featList)
        newEntorpy = 0.0
        print 'spilt feature ', i
        # 对每个value进行划分
        for value in uniqueVals:
            subDataSet = spiltDataSet(dataSet,i,value)
            print 'subDataSet:' , subDataSet
            prob = len(subDataSet)/float(len(dataSet))
                # 计算该特征进行划分后的信息
            newEntorpy += prob * calcShannonEnt(subDataSet)
            # 计算该特征进行划分后的信息增益
        infoGain = baseEntropy - newEntorpy
            # 寻找信息增益最大的特征就是最好划分的特征
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    print 'BestFeatureToSplit' 
    print bestFeature
    return bestFeature
# 投票表决，获取概率最大的那个分类
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():classCount[vote] = 0
        classCount[vote] += 1
# 反向排序，也就是从大到小
    sortedClassCount = sorted(classCount.iteritems(),key = operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]
def createTree(dataSet,labels):
    # 获取每个样本数据的分类
    classList = [example[-1] for example in dataSet]
        # 第一个迭代终止条件：选取第一个classList中的类，判断它出现的次数是否与
        # classList的长度相等，如果相等，证明classList中的所有类已经被分为同一种类
        # 则返回该类
    if classList.count(classList[0]) == len(classList):
        return classList[0]
        #如果所有特征都迭代完了，此时还没返回，
        # 说明当前仍然不能讲数据集划分为仅包含唯一类别的分组
        # 此时则返回次数出现最多的那个类别最为当前数据集的类别
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
        # 寻找最优的划分特征
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
        # 获取最优特征的各个属性
    featValues = [example[bestFeat] for example in dataSet]        
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
                # 迭代创建决策树
        myTree[bestFeatLabel][value] = createTree(spiltDataSet(dataSet,bestFeat,value),subLabels)
    return myTree
