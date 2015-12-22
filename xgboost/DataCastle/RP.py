#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Filename: RP.py.py
@Author: yew1eb
@Date: 2015/12/22 0022
"""

import xgboost as xgb
import pandas as pd
import numpy as np
from scipy import sparse

def get_model():
    X = pd.read_csv('strain_x.csv')
    Y = pd.read_csv('strain_y.csv')
    ft = pd.read_csv('sfeatures_type.csv')
    fn_cat = pd.Series(ft[ft.type=='category'].feature)
    fn_num = pd.Series(ft[ft.type=='numeric'].feature)
    x_add = pd.DataFrame([], index=X.index)
    for f in fn_cat:
        levels = np.unique(X[f])
        #print(levels)
        f_col = X.loc[:,f]
        #print(f_col)
        for level in levels:
            new_col = f_col.apply(lambda x:1 if x==level else 0)
            col_name = f +'_' + np.str(level)
            x_add[col_name] = new_col

    new_X = pd.concat([X.loc[:,'uid'], X.loc[:,fn_num], x_add], axis=1)
    #coo_mat = sparse.coo_matrix(new_X, dtype = np.float32)
    train_data = xgb.DMatrix(data=new_X.iloc[:,1:], label=1-Y.y)
    model=xgb.train(data=train_data)
    pred = 1 - model.

if __name__ == '__main__':
    get_model()