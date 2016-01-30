推荐《统计学习方法》第2章学习笔记：[K近邻法 | 码农场](http://www.hankcs.com/ml/k-nearest-neighbor-method.html)
[机器学习中的相似性度量](http://www.17bigdata.com/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E4%B8%AD%E7%9A%84%E7%9B%B8%E4%BC%BC%E6%80%A7%E5%BA%A6%E9%87%8F.html)  
本文主要记录《机器学习实战》第2章K-近邻算法中两个应用实例的学习过程。
## k-近邻算法概述
简单地说，k-近邻算法采用测量不同特征值之间的距离方法进行分类。
> **k-近邻算法**
> 
> 优点：精度高、对异常值不敏感、无数据输入假定。
> 缺点：计算复杂度高、空间复杂度高
> 适用数据范围：数值型和标称型

> **k-近邻算法的一般流程**
> 收集数据：可以使用任何方法。 
> 准备数据：距离计算所需要的数值，最好是结构化的数据格式。 
> 分析数据：可以使用任何方法。 
> 测试算法：计算错误率。 
> 使用算法：首先需要输入样本数据和结构化的输出结果，然后运行k-近邻算法判定输入数据分别属于哪个分类。

> **k-近邻算法的伪代码**
> 
> 对未知类型属性的数据集中的每个点依次执行以下操作：
> (1) 计算已知类别数据集中的点与当前点之间的距离；
> (2) 按照距离增序排序；
> (3) 选取与当前点距离最近的k个点；
> (4) 决定这k个点所属类别的出现频率；
> (5) 返回前k个点出现频率最高的类别作为当前点的预测分类。

## 使用k-近邻算法改进约会网站的配对效果
我的朋友海伦一直使用在线约会网站寻找适合自己的约会对象。尽管约会网站会推荐不同的人选，但她并不是喜欢每一个人。经过一番总结，她发现曾交往过三种类型的人：
+ 不喜欢的人
+ 魅力一般的人
+ 极具魅力的人

海伦希望我们的分类软件可以更好地帮助她将匹配对象划分到确切的分类中。此外海伦还收集了一些约会网站未曾记录的数据信息，她认为这些数据更有助于匹配对象的归类。
### 数据
数据存放在文本文件datingTestSet.txt中，每个样本数据占据一行，总共有1000行。
海伦的样本主要包含以下3种特征：
*   每年获得的飞行常客里程数
*   玩视频游戏所耗时间百分比
*   每周消费的冰淇淋公升数

```
def get_data(filename):
    mp = {'didntLike':1, 'smallDoses':2, 'largeDoses':3}
    data = pd.read_table(filename, header=None, names=['x1','x2','x3','label'])
    feature_data = data.loc[:, ['x1','x2','x3']]
    label_data = data.label.replace(mp.keys(), mp.values())
    return feature_data, label_data
```

### 归一化
为了防止特征值数量的差异对预测结果的影响(比如计算距离，量值较大的特征值影响肯定很大)，我们将所有的特征值都归一化到[0,1]。
$$newValue = \frac{oldValue - min}{max - min}$$
```
def auto_norm(data):
    min_value = data.min(0)
    max_value = data.max(0)
    ranges = max_value - min_value
    norm_data = np.zeros(data.shape)
    m = data.shape[0]
    norm_data = data - np.tile(min_value, (m,1))
    norm_data = norm_data / np.tile(ranges, (m,1))
    return norm_data
```

### 交叉验证

### 预测

## 手写识别系统
这里可以将手写字符看做由01组成的32*32个二进制文件，然后转换为1*1024的向量即为一个训练样本，每一维即为一个特征值
### 1 将一个32*32的二进制图像转换成1*1024的向量
```
def img2vector(filename):
    returnVect = np.zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect
```
### 2 手写识别系统测试代码
```
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
        
```
