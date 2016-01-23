## 关于logistic回归
我以前的笔记[Coursera机器学习课程笔记(3) Logistic Regression](http://blog.csdn.net/yew1eb/article/details/48222545)  
《统计学习方法》第6章  
以及美团技术团队的一篇文章 [Logistic Regression 模型简介](http://tech.meituan.com/intro_to_logistic_regression.html)   
[机器学习算法与Python实践之（七）逻辑回归（Logistic Regression）](http://blog.csdn.net/zouxy09/article/details/20319673)  
sklearn库中的logistic回归参数说明：
[sklearn.linear_model.LogisticRegression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression)   
[logistic regression](https://en.wikipedia.org/wiki/Logistic_regression)

logistic回归：从生产到使用
http://diyitui.com/content-1410499384.16932008.html
http://diyitui.com/content-1411104161.16932007.html
[并行逻辑回归](http://blog.sina.com.cn/s/blog_6cb8e53d0101oetv.html)
[逻辑回归 - super](http://blog.csdn.net/zc02051126/article/details/8655010)
[Softmax算法：逻辑回归的扩展 - super](http://blog.csdn.net/zc02051126/article/details/9866347)
[逻辑回归](http://blog.csdn.net/pakko/article/details/37878837)
你应该掌握的七种回归技术  http://www.csdn.net/article/2015-08-19/2825492

[【机器学习笔记1】Logistic回归总结 - 洞庭之子的专栏](http://blog.csdn.net/dongtingzhizi/article/details/15962797)

## 基本思想
Logistic回归可以用于二分类
通过，Sigmoid函数进行分类，sigmoid函数取值范围0~1,默认以0.5为界，小于0.5的被归为0类，大于0.5的被归为1类。
一直循环的找最佳的weight向量，直到不出现误分类。
如何找到最佳的weight向量呢？
梯度下降法和拟牛顿法是两种常用的方法。
具体证明可以学习《统计学习方法》



