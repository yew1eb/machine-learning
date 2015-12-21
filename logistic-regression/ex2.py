#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Filename: ex2.py
@Author: yew1eb
@Date: 2015/12/20 0020
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve, roc_curve, auc

# 导入数据集（数据来自Coursea机器学习第二周的作业ex2）
file_name = 'ex2_data.csv'
data = pd.read_csv(file_name, header=None, names=['score1', 'score2', 'result'])
score_data = data.loc[:, ['score1', 'score2']]
result_data = data.result

# 训练模型
p = 0
for i in range(10):
    x_train, x_test, y_train, y_test = train_test_split(score_data, result_data, train_size=0.7)
    model = LogisticRegression(C=1e9)
    model.fit(x_train, y_train)
    predict_y = model.predict(x_test)
    p += np.mean(predict_y == y_test)

# 绘制图像
pos_data = data[data.result == 1].loc[:, ['score1', 'score2']]
neg_data = data[data.result == 0].loc[:, ['score1', 'score2']]

h = 0.02
x_min, x_max = score_data.loc[:, ['score1']].min() - .5, score_data.loc[:, ['score1']].max() + .5
y_min, y_max = score_data.loc[:, ['score2']].min() - .5, score_data.loc[:, ['score2']].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

# 绘制边界和散点
Z = Z.reshape(xx.shape)
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
plt.scatter(x=pos_data.score1, y=pos_data.score2, color='blue', marker='o')
plt.scatter(x=neg_data.score1, y=neg_data.score2, color='red', marker='*')

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.show()

# 模型表现
answer = model.predict_proba(x_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, answer)
report = answer > 0.5
print(classification_report(y_test, report, target_names=['neg', 'pos']))
print("average precision:", p / 100)
