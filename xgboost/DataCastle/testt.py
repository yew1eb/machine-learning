#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Filename: testt.py.py
@Author: yew1eb
@Date: 2015/12/22 0022
"""

import xgboost as xgb
import pandas as pd
import numpy as np
from scipy import sparse


def get_data():
    X = pd.read_csv('strain_x.csv')
    Y = pd.read_csv('strain_y.csv')
    test = pd.read_csv('stest_x.csv')
    ft = pd.read_csv('sfeatures_type.csv')
    fn_cat = pd.Series(ft[ft.type == 'category'].feature)
    fn_num = pd.Series(ft[ft.type == 'numeric'].feature)
    x_add = pd.DataFrame([], index=X.index)
    test_add = pd.DataFrame([], index=test.index)
    for f in fn_cat:
        levels = np.unique(X[f])
        x_col = X.loc[:, f]
        test_col = test.loc[:, f]
        for level in levels:
            col_name = f + '_' + np.str(level)
            new_x_col = x_col.apply(lambda x: 1 if x == level else 0)
            x_add[col_name] = new_x_col
            new_test_col = test_col.apply(lambda x: 1 if x == level else 0)
            test_add[col_name] = new_test_col

    new_X = pd.concat([X.loc[:, 'uid'], X.loc[:, fn_num], x_add], axis=1)
    new_test = pd.concat([test.loc[:, 'uid'], test.loc[:, fn_num], test_add], axis=1)
    # coo_mat = sparse.coo_matrix(new_X, dtype = np.float32)
    dtrain = xgb.DMatrix(data=new_X.iloc[:, 1:], label=1 - Y.y)
    dtest = xgb.DMatrix(data=new_test.iloc[:, 1:])

    return dtrain, dtest


def main():

    dtrain, dtest = get_data()
    param = {
        'objective': 'binary:logistic',
        'scale_pos_weight':8.7,
        'max_depth': 2,
        'eta': 0.03,
        'gamma':0,
        'subsample':0.8,
        'lambda':1000,
        'alpha':800,
        'max_delta_step':0,
        'colsample_bytree':0.30,
        'min_child_weight':5,
        'eval_metric':'auc'
    }

    watchlist  = [(dtest,'eval'), (dtrain,'train')]
    num_round = 2
    bst = xgb.train(param, dtrain, num_round, watchlist)

def test():
    data = np.random.rand(5,10) # 5 entities, each contains 10 features
    label = np.random.randint(2, size=5) # binary target
    dtrain = xgb.DMatrix( data, label=label)
    dtest = xgb.DMatrix(test)
    param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic' }
    evallist  = [(dtest,'eval'), (dtrain,'train')]
    num_round = 10
    bst = xgb.train( param, dtrain, num_round, evallist)


if __name__ == '__main__':
    #main()
    test()
