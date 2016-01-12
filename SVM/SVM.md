
# 概念
* 超平面 ![...](http://mmbiz.qpic.cn/mmbiz/gPtPSmYD36ibq7l40SVicbgg9QibrVzThpRR43rezbgxXFN3V9jvic9wWpicAJic1uX9zF7HvuiaWPyKBnL9pcevmVGqQ/640)
* 支持向量（Support Vector）：就是离分隔超平面最近的那些点。
* 函数间隔 ![...](http://mmbiz.qpic.cn/mmbiz/gPtPSmYD369JsxlJDEsvAicHcEAhoicafX8CRWlpK3ISgZ6ZONVrcJ26pbMffwia9k7QXrRYN0NjDkaz2m6IRv3WA/640) 
* 几何间隔 ![...](http://mmbiz.qpic.cn/mmbiz/gPtPSmYD369JsxlJDEsvAicHcEAhoicafXjfPyDnibTaPOcOMG3ykzq4cgDwDjDPFk3Picyj8PzETt6U6qvQ8CFsOQ/640)

### 最大间隔分类器  
① 用几何间隔来衡量点到超平面之间的距离   
![...](http://mmbiz.qpic.cn/mmbiz/gPtPSmYD369JsxlJDEsvAicHcEAhoicafXb4Cp5Xl3AxlZ7VfHGnctAFzKDG0z5aBuRrDcJicNjZZamxeOxt0Fdww/640)  
② 一组样本中，将距超平面最近的点到超平面的距离，作为这组样本到超平面的距  
![...](http://mmbiz.qpic.cn/mmbiz/gPtPSmYD369JsxlJDEsvAicHcEAhoicafXehH4gOgXYGTQ4aeiaeDCx17A9AwW8mbf4Muxor40YQIyxwaNgMXSEaQ/640)  
③ 寻找的那个超平面，是可以使该组样本到它的距离最大，即能最好的讲样本分开  
![...](http://mmbiz.qpic.cn/mmbiz/gPtPSmYD369JsxlJDEsvAicHcEAhoicafXQ98Icr4kN5ZdHzpYg1W0cQCWF24cDHLCQFFgQtuUflG1Wb68zhFCicg/640)  
![...](http://mmbiz.qpic.cn/mmbiz/gPtPSmYD369JsxlJDEsvAicHcEAhoicafXCEiaTib0X4KS0n8iczmIQhOebGe93wPicj6Aq0Hrx9TKr3KAORYcwg8kyg/640)  
问题转化成了一个求极大值的数学问题，为了方便推导价和优化的目的，我们可以令函数间隔|f(x)|=yf(x)=1，即固定函数间隔的值为1
那么③就变成了![...](http://mmbiz.qpic.cn/mmbiz/gPtPSmYD369JsxlJDEsvAicHcEAhoicafXHh8u6qIdNWGliaPbE5hxoBdPp3oZt99oG85XomdjHvPRzUYrUvtglibA/640)，求![...](http://mmbiz.qpic.cn/mmbiz/gPtPSmYD369JsxlJDEsvAicHcEAhoicafXgpyX4HzFic6zcrVLXbnkzmBh7khyXRDkZcpGGh7HBBj73J72OQHcPzQ/640)的最大值相当于求![...](http://mmbiz.qpic.cn/mmbiz/gPtPSmYD369JsxlJDEsvAicHcEAhoicafX5RkJrL4eeBf5ONSIpR9adWfoSAodicdicuFyqmeeniamzy3MgLGjvFsibg/640)的最小值，那么最终问题进一步等价转化为了一个凸二次规划（convex quadratic programming）问题：  
![...](http://mmbiz.qpic.cn/mmbiz/gPtPSmYD369JsxlJDEsvAicHcEAhoicafXjLCZGxJLs5Ftic90FEo3tcbPtqdbeaicY4DMNGvBKKGEuVLUw9nexW8Q/640)

# 参考资料  
## 《分类战车SVM》系列 |  数说工作室
* 第一话：开题话  <http://t.cn/RAl1Qou>  
* 第二话：线性分类  <http://t.cn/RAl1QoT>
* 第三话：最大间隔分类器  <http://t.cn/RAl1Qom>

* 第四话：拉格朗日对偶问题（原来这么简单！）  <http://t.cn/RAl1QoE>
* 第五话：核函数（哦，这太神奇了！）  <http://t.cn/RAl1Qon>
* 第六话：SMO算法（像Smoke一样简单！）  <http://t.cn/RAl1QoQ>
* 附录：用Python做SVM模型   <http://t.cn/RAl1QoR>


## 支持向量机系列 | pluskid  
### 基本篇：
* [Maximum Margin Classifier](http://blog.pluskid.org/?p=632) —— 支持向量机简介。
* [Support Vector](http://blog.pluskid.org/?p=682) —— 介绍支持向量机目标函数的 dual 优化推导，并得出“支持向量”的概念。
* [Kernel](http://blog.pluskid.org/?p=685) —— 介绍核方法，并由此将支持向量机推广到非线性的情况。
* [Outliers](http://blog.pluskid.org/?p=692) —— 介绍支持向量机使用松弛变量处理 outliers 方法。
* [Numerical Optimization](http://blog.pluskid.org/?p=696) —— 简要介绍求解求解 SVM 的数值优化算法。
### 番外篇：
* [Duality](http://blog.pluskid.org/?p=702) —— 关于 dual 问题推导的一些补充理论。
* [Kernel II](http://blog.pluskid.org/?p=723) —— 核方法的一些理论补充，关于 Reproducing Kernel Hilbert Space 和 Representer Theorem 的简介。
* Regression —— 关于如何使用 SVM 来做 Regression 的简介。

* [机器学习中的算法(2)-支持向量机(SVM)基础](http://www.cnblogs.com/LeftNotEasy/archive/2011/05/02/basic-of-svm.html)  

* [SVM 的简要推导过程](http://dataunion.org/12001.html)  