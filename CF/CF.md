[协同过滤算法](http://segmentfault.com/a/1190000004022134)  
[《集体智慧编程》读书笔记](http://rudy-zhang.me/2015/12/06/%E9%9B%86%E4%BD%93%E6%99%BA%E6%85%A7%E7%BC%96%E7%A8%8B%E8%AF%BB%E4%B9%A6%E7%AC%94%E8%AE%B0/)   
[Google Python 风格指南 - 中文版](http://zh-google-styleguide.readthedocs.org/en/latest/google-python-styleguide/)
[Spark MLlib中的协同过滤 - 推酷](http://www.tuicool.com/articles/fANvieZ)   
## 协同过滤
协同过滤是利用集体智慧的一个典型方法。要理解什么是协同过滤 (Collaborative Filtering, 简称 CF)，首先想一个简单的问题，如果你现在想看个电影，但你不知道具体看哪部，你会怎么做？大部分的人会问问周围的朋友，看看最近有什么好看的电影推荐，而我们一般更倾向于从口味比较类似的朋友那里得到推荐。这就是协同过滤的核心思想。
协同过滤一般是在海量的用户中发掘出一小部分和你品位比较类似的，在协同过滤中，这些用户成为邻居，然后根据他们喜欢的其他东西组织成一个排序的目录推荐给你。
要实现协同过滤，需要以下几个步骤：
1. 搜集偏好
2. 寻找相近用户
3. 推荐物品


## 寻找相近用户（相似度计算）
收集完用户信息后，我们通过一些方法来确定两个用户之间品味的相似程度，计算他们的相似度评价值。有许多计算方法，例如：欧几里得距离、皮尔逊相关度、曼哈顿距离、Jaccard系数等等。
### 欧几里得距离
欧几里德距离（Euclidean Distance），最初用于计算欧几里得空间中两个点的距离，在二维空间中，就是我们熟悉的两点间的距离，x、y表示两点，维度为n：
$$d(x,y)=\sqrt {(\sum_i^n (x_i-y_i)^2)}$$
相似度：$$sim(x,y)={1\over {1+d(x,y)}}$$

### 皮尔逊相关度
[皮尔逊相关系数](https://zh.wikipedia.org/wiki/%E7%9A%AE%E5%B0%94%E9%80%8A%E7%A7%AF%E7%9F%A9%E7%9B%B8%E5%85%B3%E7%B3%BB%E6%95%B0)（Pearson Correlation Coefficient），用于判断两组数据与某一直线拟合程度的一种度量，取值在[-1,1]之间。当数据不是很规范的时候（如偏差较大"夸大分值"），皮尔逊相关度会给出较好的结果。
$$p(x,y)={{\sum {x_iy_i}-n\overline{xy}}\over {(n-1)S_xS_y}}={{n\sum {x_iy_i}-\sum x_i\sum y_i}\over{\sqrt{n\sum{x_i^2}-(\sum x_i)^2}{\sqrt{n\sum{y_i^2}-(\sum y_i)^2}}}}$$

### 曼哈顿距离
曼哈顿距离（Manhattan distance），就是在欧几里得空间的固定直角坐标系上两点所形成的线段对轴产生的投影的距离总和。
$$d(x,y)={\sum{\|x_i-y_i\|}}$$

#### Jaccard系数
Jaccard系数，也称为Tanimoto系数，是Cosine相似度的扩展，也多用于计算文档数据的相似度。通常应用于x为布尔向量，即各分量只取0或1的时候。此时，表示的是x,y的公共特征的占x，y所占有的特征的比例:
$$T(x,y)={x\bullet y\over{\|x\|^2}+\|y\|^2-x\bullet y}={\sum{x_iy_i}\over{\sqrt{\sum{x_i^2}}}+\sqrt{\sum{y_i^2}}-\sum{x_iy_i}}$$

### 为评论者打分
到此，我们就可以根据计算出用户之间的相关度，并根据相关度来生成相关度列表，找出与用户口味相同的其他用户。
```python
#推荐用户
def topMatches(prefs,person,n=5,similarity=sim_distance):
    #python列表推导式
    scores=[(similarity(prefs,person,other),other) for other in prefs if other!=person]
    scores.sort()
    scores.reverse()
    return scores[0:n]
```

<http://segmentfault.com/a/1190000004022134>
<http://www.cnblogs.com/mdyang/archive/2011/07/09/PCI-ch2.html>
<http://rudy-zhang.me/2015/12/06/%E9%9B%86%E4%BD%93%E6%99%BA%E6%85%A7%E7%BC%96%E7%A8%8B%E8%AF%BB%E4%B9%A6%E7%AC%94%E8%AE%B0/>
<http://kuroro.me/lou/ML/PCI/cap2/cap2.html>
