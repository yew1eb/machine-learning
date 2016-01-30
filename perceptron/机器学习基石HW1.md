记录一下《机器学习基石》作业一中与感知机相关的练习，其中第15-20题涉及到naive、random sample和pocket的感知机算法。
### Question 15-17
DATA: <https://d396qusza40orc.cloudfront.net/ntumlone%2Fhw1%2Fhw1_15_train.dat>
Each line of the data set contains one (xn,yn) with xn∈R4\. T 
he first 4 numbers of the line contains the components of xn orderly, the last number is yn.
 Please initialize your algorithm with w=0 and take sign(0) as −1 
#### Question 15
Implement a version of PLA by visiting examples in the naive cycle using the order of examples in the data set. Run the algorithm on the data set. What is the number of updates before the algorithm halts?
输入有4个维度，输出为{-1，+1}。共有400条数据。
题目要求将权向量元素初始化为0，然后使用“Naive Cycle”遍历训练集，求停止迭代时共对权向量更新了几次。
所谓“Naive Cycle”指的是在某数据条目x(i)上发现错误并更新权向量后，下次从x(i+1)继续读数据，而不是回到第一条数据x(0)从头开始。

#### Question 16
random打乱输入样本顺序，然后在这轮计算中始终使用这一排序，直到下一轮计算开始再重新排序，重复2000次，求对权向量的平均修正次数。

#### Question 17
在Question 16的基础上，修改alpha学习率为0.5
```
import sys
import random
import numpy as np

def load_data() :
    X = []
    Y = []
    m = []
    with open('hw1_15_train.dat') as f :
        for line in f :
            m.append([float(i) for i in line.split()])
    mm = np.array(m)
    row = len(m)
    X = np.c_[np.ones(row), mm[::, :-1]]
    Y = mm[:,-1]
    return X, Y
    
def sign(x) :
    if x > 0 :
        return 1.
    return -1.

def train(X, Y, rand = False, alpha = 1) :
    col = len(X[0])
    n = len(X)
    w = np.zeros(col)
    ans = 0
    idx = range(n)
    if rand :
        idx = random.sample(idx, n)
    k = 0
    update = False
    while True :
        i = idx[k]
        if sign(np.dot(X[i], w)) != Y[i] :
            ans += 1
            w = w + alpha * Y[i] * X[i, :]
            update = True
        k += 1
        if k == n :
            if update == False :
                break
            k = 0
            update = False
    return ans

def naive_cycle() :
    X, Y = load_data()
    ans = train(X, Y)
    print ans

def predefined_random(n, alpha = 1) :
    X, Y = load_data()
    cnt = 0
    for i in xrange(n) :
        cnt += train(X, Y, rand = True, alpha = alpha)
    print cnt / n

def main() :
    naive_cycle()
    predefined_random(2000)
    predefined_random(2000, 0.5)

if __name__ == '__main__' :
    main()
```
### Question 18-20
Train DATA: <https://d396qusza40orc.cloudfront.net/ntumlone%2Fhw1%2Fhw1_18_train.dat>
 Test DATA: <https://d396qusza40orc.cloudfront.net/ntumlone%2Fhw1%2Fhw1_18_test.dat>

#### Question 18
As the test set for "verifying" the g returned by your algorithm (see lecture 4 about verifying). The sets are of the same format as the previous one.
Run the pocket algorithm with a total of 50 updates on D, and verify the performance of w using the test set.Please repeat your experiment for 2000 times, each with a different random seed.What is the average error rate on the test set?
第18题要求在第16题 Random PLA 算法的基础上使用 Pocket 算法对数据做二元划分。Pocket算法在[第2篇文章](http://my.oschina.net/findbill/blog/205805)介绍过，通常用来处理有杂质的数据集，在每一次更新 Weights（权向量）之后，把当前犯错最少的Weights放在pocket中，直至达到指定迭代次数(50次)，pocket中的Weights即为所求。然后用测试数据验证W(pocket)的错误率，进行2000次计算取平均。
简而言之就是，“pocket不影响pla的正常运行，每轮W该更新还是要更新；pocket只需要维护历史出现的W中，在train_data上error最小的那个即可”

#### Question 19
Modify your algorithm in Question 18 to return w50 (the PLA vector after 50 updates) instead of Wg (the pocket vector) after 50 updates. Run the modified algorithm on D, and verify the performance using the test set. Please repeat your experiment for 2000 times, each with a different random seed. What is the average error rate on the test set? 
题19要求用经过50次更新的W(50)进行验证，而不是W(pocket)，由于W(50)未必是当下最优，所以平均错误率一定会升高。代码几乎没有改动，只需在调用 getTestErrRate 函数是传入W(50)的指针即可。
#### Question 20
Modify your algorithm in Question 18 to run for 100 updates instead of 50, and verify the performance  of wPOCKET using the test set. Please repeat your experiment for 2000 times, each with a different random seed.  What is the average error rate on the test set? 
本题要求把 Weights 的更新次数从50增加到100，可以预计平均错误率是降低的。
```
import sys
import random
import numpy as np

def load_data(file_path) :
   X = []
   Y = []
   m = []
   with open(file_path) as f :
      for line in f :
         m.append([float(i) for i in line.split()])
   mm = np.array(m)
   row = len(m)
   X = np.c_[np.ones(row), mm[::, :-1]]
   Y = mm[:,-1]
   return X, Y

def sign(x) :
   if x > 0 :
      return 1.
  return -1.

def test(X, Y, w) :
   n = len(Y)
   ne = sum([1 for i in range(n) if sign(np.dot(X[i], w)) != Y[i]])
   return ne / float(n)

def train(X, Y, updates = 50, pocket = True) :
   col = len(X[0])
   n = len(X)
   w = np.zeros(col)
   wg = w
   error = test(X, Y, w);

   for k in range(updates) :
      idx = random.sample(range(n), n)
      for i in idx :
         if sign(np.dot(X[i], w)) != Y[i] :
            w = w + Y[i] * X[i]
            e = test(X, Y, w)
            if e < error :
               error = e
               wg = w
            break
 if pocket :
      return wg
   return w

def main() :
   X, Y = load_data('hw1_18_train.dat')
   TX, TY = load_data('hw1_18_test.dat')
   error = 0
  n = 200
  for i in range(n) :
      #w = train(X, Y, updates = 50)
 #w = train(X, Y, updates = 50, pocket = False)  w = train(X, Y, updates = 100, pocket = True)
      error += test(TX, TY, w)
   print(error / n)

if __name__ == '__main__' :
   main()
```