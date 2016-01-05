from numpy import *
import operator

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    lables = ['A','A','B','B']
    return group,lables

# KNN 分类算法
def classify0(inx,dataSet,labels,k):
    dataSetSize = dataSet.shape[0] # shape[0]获取行 shape[1] 获取列
    # 第一步，计算欧式距离
    diffMat = tile(inx,(dataSetSize,1)) - dataSet  #tile类似于matlab中的repmat，复制矩阵
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distance = sqDistances ** 0.5
    sortedDistIndecies = distance.argsort()  # 增序排序
    classCount = {}
    for i in range(k):
    # 获取类别 
        voteIlabel = labels[sortedDistIndecies[i]]
        #字典的get方法，查找classCount中是否包含voteIlabel，是则返回该值，不是则返回defValue，这里是0
        # 其实这也就是计算K临近点中出现的类别的频率，以次数体现
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
        #  对字典中的类别出现次数进行排序，classCount中存储的事 key-value，其中key就是label，value就是出现的次数
        #  所以key=operator.itemgetter(1)选中的事value，也就是对次数进行排序
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
        #sortedClassCount[0][0]也就是排序后的次数最大的那个label
    return sortedClassCount[0][0]
