## scikit-learn介绍
scikit-learn是Python的一个开源机器学习模块，它建立在NumPy，SciPy和matplotlib模块之上。值得一提的是，scikit-learn最先是由David Cournapeau在2007年发起的一个Google Summer of Code项目，从那时起这个项目就已经拥有很多的贡献者了，而且该项目目前为止也是由一个志愿者团队在维护着。  
scikit-learn主页：[scikit-learn homepage](scikit-learn.org/dev/)   
主要包含以下部分：  

* [分类算法](http://scikit-learn.org/stable/supervised_learning.html#supervised-learning)
* [回归分析](http://scikit-learn.org/stable/supervised_learning.html#supervised-learning)
* [聚类算法](http://scikit-learn.org/stable/modules/clustering.html#clustering)
* [降维算法](http://scikit-learn.org/stable/modules/decomposition.html#decompositions)
* [模型选择](http://scikit-learn.org/stable/model_selection.html#model-selection)
* [数据预处理](http://scikit-learn.org/stable/modules/preprocessing.html#preprocessing)

## 模型选择
### 交叉验证(Cross-validation: evaluating estimator performance)
#### S-fold交叉检验
应用最多的是S折交叉检验(S-fold cross validation)，方法如下：首先随机地将已给数据切分为S个互不相交的大小相同的子集；然后利用S-1个子集的数据训练模型，利用余下的子集测试模型；将这一过程对可能的S种选择重复进行；最后选出S次评测中平均测试误差最小的模型。   
#### leave-one-out交叉检验方法
留一交叉检验(leave-one-out cross validation)是S折交叉检验的特殊情形，是S为给定数据集的容量时情形。
我们可以从训练数据中挑选一个样本，然后拿其他训练数据得到模型，最后看该模型是否能将这个挑出来的样本正确的分类。

### 格点搜索(Grid Search: Searching for estimator parameters)
#### [Exhaustive Grid Search](http://scikit-learn.org/stable/modules/grid_search.html#exhaustive-grid-search)
#### [randomized-parameter-optimization](http://scikit-learn.org/stable/modules/grid_search.html#randomized-parameter-optimization)

## 数据预处理
### [干货：结合Scikit-learn介绍几种常用的特征选择方法](http://www.tuicool.com/articles/ieUvaq)
### 数据归一化(Data Normalization)
归一化：把数变为（0，1）之间的小数  
标准化：将数据按比例缩放，使之落入一个小的特定区间  

    from sklearn import preprocessing  
    # normalize the data attributes  
    normalized_X = preprocessing.normalize(X)
    # standardize the data attributes
    standardized_X = preprocessing.scale(X)

## 参考资料 
* [Scikit Learn: 在python中机器学习](http://my.oschina.net/u/175377/blog/84420)
* [干货：结合Scikit-learn介绍几种常用的特征选择方法](http://www.tuicool.com/articles/ieUvaq)
* [python并行调参——scikit-learn grid_search](http://blog.csdn.net/abcjennifer/article/details/23884761)
* [Newest'scikit-learn' Questions -Stack Overflow](http://stackoverflow.com/questions/tagged/scikit-learn)