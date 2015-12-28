#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Filename: id3.py
@Author: yew1eb
@Date: 2015/12/24 0021
"""
import numpy as np

def load_data():
    '''
    导入数据集
    '''
    dataset = [[1, 2, 0, 1, 0, 'yes'],
         [0, 1, 1, 0, 1, 'yes'],
         [1, 0, 0, 0, 1, 'no'],
         [2, 1, 1, 0, 1, 'no'],
         [1, 1, 0, 1, 1, 'no']]

    return dataset

def calc_entropy(dataset):
    '''
    计算熵
    '''
    num = dataset.shape[0]
    # 统计y中不同label值的个数
    label_count = {}
    for label in dataset:
        label = dataset
        if label not in label_count:
            label_count[label] = 0
        label_count[label] += 1
    entroy = 0.0
    for key in label_count:
        prob = float(label_count[key])/num
        entroy -= prob * np.log2(prob)
    return entroy

def split_dataset(X, y, index, value):
    '''
    划分数据集
    返回数据集中特征下标为index，特征值等于value的子数据集
    '''
    ret = []

