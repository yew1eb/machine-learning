[Linear Regression总结](http://blog.csdn.net/dongtingzhizi/article/details/16884215)
[linear-regression](http://muxuezi.github.io/posts/2-linear-regression.html)
在探究线性回归模型之前了解一下线性回归模型的学习步骤：
![vlcsnap-2014-09-23-23h30m27s24](http://dataaspirant.files.wordpress.com/2014/09/vlcsnap-2014-09-23-23h30m27s241.png?w=1000)
![vlcsnap-2014-09-24-20h44m07s18](http://dataaspirant.files.wordpress.com/2014/09/vlcsnap-2014-09-24-20h44m07s18.png?w=1000)


线性回归：
线性回归(Linear Regression)：
优点：结果易于理解，计算上不复杂。
缺点：对非线性数据拟合不好。
适用数据类型：数值型和标称型数据。
算法类型：回归算法。
ps:回归于分类的不同，就在于其目标变量时连续数值型。

简述：在统计学中，线性回归（Linear Regression）是利用称为线性回归方程的最小平方函数对一个或多个自变量和因变量之间关系进行建模的一种回归分析。这种函数是一个或多个称为回归系数的模型参数的线性组合（自变量都是一次方）。只有一个自变量的情况称为简单回归，大于一个自变量情况的叫做多元回归。

线性方程的模型函数的向量表示形式为：
![..](http://jbcdn2.b0.upaiyun.com/2014/08/89047e4648479810179898af1a338ef9.png)
 

通过训练数据集寻找向量系数的最优解，即为求解模型参数。其中求解模型系数的优化器方法可以用“最小二乘法”、“梯度下降”算法，来求解损失函数：

![..](http://jbcdn2.b0.upaiyun.com/2014/08/4d81c350617b27715a75da2c9a09a118.png)
的最优值。

四、回归算法
    回归算法主要处理连续型的问题，主要的算法有如下的一些：
1、基本的线性回归与局部加权线性回归
    地址(http://blog.csdn.net/google19890102/article/details/26074827)，在这个方面主要实现了基本的线性回归和局部加权线性回归。主要通过正规方程组的方式求解权重。在线性回归中最容易出现过拟合和欠拟合的问题，所以引入局部加权线性回归，通过权重调节每个特征的重要程度。
2、基本线性回归的另类实现
    地址(http://blog.csdn.net/google19890102/article/details/26616973)，在这里可以使用广义逆求解其权重，这也是在ELM中学到的一点。
3、岭回归
    地址(http://blog.csdn.net/google19890102/article/details/27228279)，岭回归主要还是在处理过拟合的问题，我们需要找到方差和偏差折中的算法，岭回归主要是通过正则项去做模型的选择。