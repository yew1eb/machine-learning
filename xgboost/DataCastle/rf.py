#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
@filename: rf.py.py
@author: yew1eb
@site: http://blog.yew1eb.net
@contact: yew1eb@gmail.com
@time: 2016/01/16 下午 10:47
'''

import pandas as pd
import numpy as np
from sklearn import cross_validation
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import time

def load_data(dummy=False):
    path = 'D:/dataset/rp/'
    train_x = pd.read_csv(path + 'train_x.csv')
    train_x = train_x.drop(['uid'], axis=1)

    train_y = pd.read_csv(path + 'train_y.csv')
    train_y = train_y.drop(['uid'], axis=1)

    test_x = pd.read_csv(path + 'test_x.csv')
    test_uid = test_x.uid
    test_x = test_x.drop(['uid'], axis=1)

    if dummy: # 将分类类型的变量转为哑变量
        features = pd.read_csv(path + 'features_type.csv')
        features_category = features.feature[features.type == 'category']
        encoded = pd.get_dummies(pd.concat([train_x, test_x], axis=0), columns=features_category)
        train_rows = train_x.shape[0]
        train_x = encoded.iloc[:train_rows, :]
        test_x  = encoded.iloc[train_rows:, :]

    return train_x, train_y, test_x, test_uid

def sklearn_random_forest(train_x, train_y, test_x, test_uid):
    # 设置参数
    clf = RandomForestClassifier(n_estimators=5,
                                 bootstrap=True, #是否有放回的采样
                                 oob_score=False,
                                 n_jobs=4, #并行job个数
                                 min_samples_split=5)
    # 训练模型
    n_samples = train_x.shape[0]
    cv = cross_validation.ShuffleSplit(n_samples, n_iter=3, test_size=0.3, random_state=0)
    predicted = cross_validation.cross_val_predict(clf, train_x, train_y, cv=cv)
    print(metrics.accuracy_score(train_y, predicted))

    test_y = clf.predict(test_x)
    result = pd.DataFrame({"uid":test_uid, "score":test_y}, columns=['uid','score'])
    result.to_csv('rf_'+str(time.time())+'.csv', index=False)

def main():
    train_x, train_y, test_x, test_uid = load_data(dummy=True)
    sklearn_random_forest(train_x, train_y, test_x, test_uid)

if __name__ == '__main__':
    main()