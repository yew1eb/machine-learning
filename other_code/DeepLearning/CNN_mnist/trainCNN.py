# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 10:59:38 2015

@author: lab-liu.longpo
"""

from cnn import funcnn
import numpy as np


def floatrange(start,stop,steps):
    return [start+float(i) * (stop-start)/(float(steps)-1) for i in range(steps)]


#LR = [0.03]
#LR = floatrange(0.033,0.036,10)
LR = [0.0015]
BS = [100]
#result = np.empty((3,8,10),dtype="float32")
k = 0;
for i in range(1):
    for j in range(1):
        print 'test',k
        k = k+1
        tmp = funcnn(LR[i],BS[j])
        #result[i,j,:] = tmp.history['val_acc']
        print 'learning rate:',LR[i],'batch size:',BS[j]
        
