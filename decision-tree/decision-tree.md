[机器学习中的算法(1)-决策树模型组合之随机森林与GBDT](http://www.cnblogs.com/LeftNotEasy/archive/2011/03/07/random-forest-and-gbdt.html)  
[决策树 | 码农场](http://www.hankcs.com/ml/decision-tree.html)  
[机器学习实战笔记3(决策树与随机森林)](http://blog.csdn.net/lu597203933/article/details/38024239)  
[维基百科-决策树学习](https://zh.wikipedia.org/wiki/%E5%86%B3%E7%AD%96%E6%A0%91%E5%AD%A6%E4%B9%A0)  
[算法杂货铺——分类算法之决策树(Decision tree)](http://www.cnblogs.com/leoo2sk/archive/2010/09/19/decision-tree.html)
[决策树 - 随机森林与GBDT](http://www.cnblogs.com/LeftNotEasy/archive/2011/03/07/random-forest-and-gbdt.html)  
[【机器学习基础】决策树算法](http://www.jianshu.com/p/32fe8746ef87) 
 [决策树学习笔记整理](http://www.cnblogs.com/bourneli/archive/2013/03/15/2961568.html?cm_mc_uid=80825002416214541407450&cm_mc_sid_50200000=1454140745)  
决策树的类型有很多，有CART、ID3和C4.5等，其中CART是基于基尼不纯度(Gini)的，这里不做详解，而ID3和C4.5都是基于信息熵的，它们两个得到的结果都是一样的,本次定义主要针对ID3算法。

### 决策树的构造

```
优点：计算复杂度不高，输出结构易于理解，对中间值的缺失不敏感，可以处理不相关特征数据。
缺点：可能会产生过度匹配问题
适用数据类型：数值型和标称型
```
决策树学习采用的是自顶向下的递归方法，其基本思想是以信息熵为度量构造一棵熵值下降最快的树，到叶子节点处的熵值为零，此时每个叶节点中的实例都属于同一类。

在构造决策树时，我们需要解决的第一个问题就是，当前数据集上哪个特征在划分数据分类时起决定性作用。为了找到决定性的特征，划分出最好的结果，我们必须评估每个特征。划分出数据子集，然后递归的进行划分，直到已经当前分支下的数据已经属于同一类型，无需进一步对数据进行分割。

决策树的一般流程
```
(1)收集数据：可以使用任何方法。
(2)准备数据：树构造算法只适用于标称型数据，因此数值型数据必须离散化。
(3)分析数据：可以使用任何方法，构造树完成之后，我们应该检查图形是否符合预期。
(4)训练算法：构造树的数据结构。
(5)测试算法：使用经验树计算错误率。
(6)使用算法：此步骤可以适用于任何监督学习算法，而使用决策树可以更好地理解数据
的内在含义。
```

### 信息增益
在划分数据集之后信息发生的变化获得的信息增益称为信息增益，而信息增益集合的度量方式称为熵。
熵定义为信息的期望值，事件Xi发生的概率为p(Xi)，那么事件Xi的信息定义为：
$$ l(x_i)=-log_2p(x_i)$$
所有可能发生事件的信息期望值，即为熵：
$$H =-\sum_1^np(x_i)log_2p(x_i)$$

## 决策树算法
### ID3
ID3的原理是基于信息熵增益达到最大，设原始问题的标签有正例和负例，p和n表示其相应的个数。则原始问题的信息熵为
![...](http://images.cnitblog.com/blog/359970/201305/28161639-b5fffb7b93c648649ac4dcc31674f4eb.jpg)

**信息增益（information gain）**：一个属性的信息增益就是由于使用这个属性分割样例而导致的期望熵降低。
![...](http://images.cnitblog.com/blog/359970/201305/28161810-649eec27e1d6458bb007d23e20842ef0.jpg)
ID3的原理即使Gain达到最大值。信息增益即为熵的减少或者是数据无序度的减少。

ID3易出现的问题：
如果是取值更多的属性，更容易使得数据更“纯”（尤其是连续型数值），其信息增益更大，决策树会首先挑选这个属性作为树的顶点。结果训练出来的形状是一棵庞大且深度很浅的树，这样的划分是极为不合理的。 此时可以采用C4.5来解决。

### C4.5
C4.5的思想是最大化Gain除以下面这个公式即得到信息增益率：
![...](http://img.blog.csdn.net/20150513143256228?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvTHU1OTcyMDM5MzM=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)
其中底为2
![info-gain](http://dataunion.org/wp-content/uploads/2015/03/info-gain.png)  
$$IG\\_ratio=\frac{IG(V)}{H(V)}$$
$$H(V)=-\sum_j{p(v_j)logp(v_j)}$$
## CART
CART 基于基尼不纯度（Gini impurity）
