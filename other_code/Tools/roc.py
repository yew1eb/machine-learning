#!/usr/local/bin/python
# -*- coding : utf-8 -*-

import os 
import sys 
import math 

"""
Author : zengkui111@gmail.com
Created Time : Sun 04 Nov 2012 09:48:56 PM CST
FileName : roc.py
ChangeLog : 
Description :  
    The input format include two columns 
        first : the label in real
        second : the probability of classification 
    The output is  the point coordinate of roc curve
"""

if __name__ == "__main__" :

    fp = sys.stdin
    instance = []
    positive = 0.0
    negative = 0.0
    while True :
        line = fp.readline()
        if len(line) <= 0 :
            break
        label,p = line.strip().split()
        label = int(label)
        p = float(p)
        if label > 0 :
            positive += 1.0
        else :
            negative += 1.0
        if p == 1 :
            p = 0.999999
        instance.append((p/(1-p), label)) 
    instance.sort()
    instance.reverse()
    tp = 0.0
    fp = 0.0
    for p,label in instance :
        if label > 0 :
            tp += 1
        else :
            fp += 1
        #output is point (x,y)
        print fp/negative,tp/positive

