#!/usr/bin/python 
# -*- coding: utf-8 -*- 

__author__ = "zengkui"
__version__ = "$Revision: 1.0 $" 
__date__ = "$Date: 2012/08/04 21:20:19 $"
__copyright__ = "Copyright (c) 2012 zengkui"
__license__ = "Python"


import os
import sys
import math

"""
A very intresting problem.
23784 = 2
36781 = 3
80312 = 3 
19032 = 3
10986 = 5
What the next result ?
30390 = ?
16023 = ?
......
This problem can be solved by linear regression.
The following code is the answer.
"""

class LinearRegression :
    def __init__ ( self ) :
        self.__LEARNING_RATE = 0.1
        self.__MAX_FEATURE_CNT = 11
        #the init weight of digit from  0 to 9  and also the Intercept
        self.__theta  = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] 
        #training sample 
        self.samples = [[1,1,0,1,0,0,1,0,0,1,1,4], \
                        [1,0,1,0,0,1,0,0,3,0,0,0], \
                        [1,0,0,2,2,0,1,0,0,0,0,0], \
                        [1,2,0,1,0,1,1,0,0,0,0,2], \
                        [1,0,1,0,0,0,0,0,3,1,0,2], \
                        [1,0,1,0,0,1,1,1,1,0,0,1], \
                        [1,0,1,0,0,1,0,1,0,1,1,4], \
                        [1,2,0,1,1,0,0,0,0,0,1,3], \
                        [1,0,1,0,0,2,0,0,0,2,0,4], \
                        [1,0,0,1,0,1,1,0,0,1,1,3], \
                        [1,0,1,0,0,0,2,0,0,1,1,3], \
                        [1,1,1,1,1,0,0,1,0,0,0,2], \
                        [1,0,0,0,0,0,0,2,1,1,1,5]]
        #test cases the last colum is answer for check
        self.test_cases = \
            [[1,1,0,0,1,1,1,0,0,1,0,3], \
            [1,1,0,1,0,1,0,1,0,0,1,3], \
            [1,0,1,1,1,0,0,1,0,1,0,3], \
            [1,0,1,0,0,0,1,1,0,2,0,5], \
            [1,0,1,0,1,1,1,0,0,1,0,2], \
            [1,1,1,0,0,0,0,1,0,1,1,5], \
            [1,0,0,1,0,0,1,0,0,0,3,3], \
            [1,0,1,0,0,1,1,1,1,0,0,1], \
            [1,0,0,0,2,0,0,0,1,0,2,2], \
            [1,0,1,0,0,3,1,0,0,0,0,0], \
            [1,0,1,1,2,0,0,0,0,1,0,2]]

    def __hypothesis ( self, x ) :
        h = 0
        for idx in range ( 0, self.__MAX_FEATURE_CNT) :
            h += x[idx] * self.__theta[idx]
        return h 

    def __update_theta (self, x, delta ) :
        for idx in range (0, self.__MAX_FEATURE_CNT) :
            self.__theta[idx] -= x[idx] * delta

    def __train (self ) :
        for x in self.samples :
            h = self.__hypothesis ( x[0:-1] )
            y = x[self.__MAX_FEATURE_CNT]
            delta = (h - y) * self.__LEARNING_RATE
            self.__update_theta ( x, delta )

    def __get_loss (self) :
        loss_sum = 0 
        for x in self.samples : 
            h = self.__hypothesis ( x[0:-1] ) 
            y = x[self.__MAX_FEATURE_CNT]
            loss_sum += (h - y) * ( h - y ) / 2 
        return loss_sum
    
    def online_training (self) :
        for itr in range (0,  100) :
            self.__train()
            loss_sum = self.__get_loss ()
            for i in range ( 0, self.__MAX_FEATURE_CNT ) :
                print "theta[%d] = %lf" % (i, self.__theta[i])
            print "The %dth iterator and loss is %lf" % ( itr, loss_sum)
            if loss_sum < 0.00001 :
                break

    def test (self) :
        for t in self.test_cases :
            h = self.__hypothesis ( t[0:-1] )
            print "H = %lf, ANS = %d" % ( h, t[self.__MAX_FEATURE_CNT])

if __name__ == "__main__" :

    lr = LinearRegression();
    lr.online_training()
    lr.test()

