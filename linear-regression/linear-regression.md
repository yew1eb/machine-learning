* [【机器学习笔记2】Linear Regression总结](http://blog.csdn.net/dongtingzhizi/article/details/16884215)  
* [sicikt-learn | linear-regression](http://muxuezi.github.io/posts/2-linear-regression.html)  
* [Coursera机器学习课程笔记(2) Linear Regression](http://blog.csdn.net/yew1eb/article/details/48213355)    
* [最小二乘法多项式曲线拟合原理与实现](http://blog.csdn.net/jairuschan/article/details/7517773)  
* [李航《统计学习方法》多项式函数拟合问题V2](http://blog.csdn.net/xiaolewennofollow/article/details/46757657)
* [Standford机器学习 线性回归Cost Function和Normal Equation的推导](http://blog.csdn.net/jackie_zhu/article/details/8883782)  
* [Andrew Ng机器学习公开课笔记 -- 线性回归和梯度下降](http://www.cnblogs.com/fxjwind/p/3626173.html)  
* [线性回归方法概要 BLOG | 逍遥郡](http://blog.jqian.net/post/linear-regression.html)  
************

## 线性回归(Linear Regression)
优点：结果易于理解，计算上不复杂  
缺点：对非线性数据拟合不好  
适用数据类型：数值型和标称型数据  
ps:回归于分类的不同，就在于其目标变量时连续数值型

线性方程的模型函数的向量表示形式为：  
![..](http://jbcdn2.b0.upaiyun.com/2014/08/89047e4648479810179898af1a338ef9.png)  
通过训练数据集寻找向量系数的最优解，即为求解模型参数。其中求解模型系数的优化器方法可以用“最小二乘法”、“梯度下降”算法，来求解损失函数：  
![..](http://jbcdn2.b0.upaiyun.com/2014/08/4d81c350617b27715a75da2c9a09a118.png)
的最优值。

## 局部加权线性回归(Locally Weighted Linear Regression, LWLR)
在线性回归中最容易出现过拟合和欠拟合的问题，所以引入局部加权线性回归，通过权重调节每个特征的重要程度。    

  
