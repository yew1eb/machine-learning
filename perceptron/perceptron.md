《统计学习方法》：感知机，笔记推荐：[感知机](http://www.hankcs.com/ml/the-perceptron.html)
## Perceptron Learning Algorithm 
[Perceptron - 感知机](http://zh.wikipedia.org/wiki/%E6%84%9F%E7%9F%A5%E5%99%A8) 它能够根据每笔资料的特征，把资料判断为不同的类别。令h(x)是一个perceptron，你给我一个x(x是一个特征向量)，把x输入h(x)，它就会输出这个x的类别，譬如在信用违约风险预测当中，输出就可能是这个人会违约，或者不会违约。本质上讲，perceptron是一种二元线性分类器,它通过对特征向量的加权求和，并把这个”和”与事先设定的门槛值(threshold)做比较，高于门槛值的输出1，低于门槛值的输出-1。
![perceptron](http://i.imgur.com/ZiLge0Z.png)  
这里的思想在于朴素的把从用户信息抽出来的一些feature（年龄等）量化并组成vector，然后乘以一个权重向量，并设定一个阈值，大于这个阈值就表示好，小于表示不好，很明显这个式子的未知变量有两个（实际只有一个）：  
权重向量 wi, 1<=i<=d  
阈值，下面设为0  
做一点小小的变形使得式子更加紧凑   
![simpify](http://i.imgur.com/Ff257eu.png)     
 还有就是从这个模型可以知道，regression model也可以解决classification问题，转化的思想。下面是这个算法的核心，定义了学习目标之后，如何学习？这里的学习是，如何得到最终的直线去区分data？  
![...](http://i.imgur.com/8LsJHL8.png)  
 这个算法的精髓之处在于如何做到"做错能改"，其循环是不断遍历feature vector，找到错误的点（Yn和当前Wt*Xn不符合），然后校正Wt，那么为什么要这样校正？以及为什么最终一定会停止？  
![...](http://i.imgur.com/urQeggf.png)    
 看起来wt+1是更接近wf了，但他们内积的增大并不能表示他们夹角的变小，也有可能是因为wt+1长度||wt+1||增大了。但是||wt+1||的增加是有限的:  
![...](http://i.imgur.com/L2fttGc.png)  
根据上面的性质，我们可以来求夹角余弦。从w0=0(初始的向量)开始，经过T次错误更正，变为wT，那么就有:    
![...](http://i.imgur.com/GMFjnwz.png)  
## Pocket Algorithm
 有时我们拿到的数据数量庞大，或是不是线性可分的，这个时候用PLA将消耗大量的时间，或是根本无法停止，这个时候我们可以使用一种委曲求全的办法，在PLA中加入pocket step。这个pocket是做什么用的呢？这个pocket会使用PLA在每次迭代中产生的w，带进原始数据，去计算分类错误率，并记录最好的那个w，譬如我们设定让PLA迭代N次就停止，则pocket返回这N次迭代中出现的最好的w。当N足够大的时候，pocket总能返回还不错的结果。