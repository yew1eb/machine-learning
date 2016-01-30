#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
@filename: add_data.py.py
@author: yew1eb
@site: http://blog.yew1eb.net
@contact: yew1eb@gmail.com
@time: 2016/01/01 下午 12:00
'''

import pandas as pd
import numpy as np

path = 'd:/dataset/rp/'
test_x_csv = path + 'test_x.csv'
test_y_csv = './0.717.csv'
dtest_x = pd.read_csv(test_x_csv)
dtest_y = pd.read_csv(test_y_csv)

test_xy = pd.merge(dtest_x, dtest_y, on='uid')
add_low  = test_xy[test_xy.score < 0.1]
add_high = test_xy[test_xy.score > 0.97]
add_test_xy = pd.concat([add_low,add_high], axis=0)

add_test_xy = add_test_xy.drop_duplicates(cols='uid')

print(add_test_xy)
add_y = add_test_xy[['uid','score']].copy()
add_y['score'] = np.where(add_y['score']<0.5, 0, 1)
add_y.columns = ['uid', 'y']
add_X = add_test_xy.drop(['score'], axis=1)

add_y.to_csv(path+'add_y.csv', index=False)
add_X.to_csv(path+'add_X.csv', index=False)