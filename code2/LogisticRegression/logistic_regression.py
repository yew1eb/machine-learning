#! /usr/bin/python 
# -*- coding: utf-8 -*-

__author__ = "zengkui"
__version__ = "$Revision: 1.0 $"
__date__ = "$Date: 2012/08/11 21:20:19 $"
__copyright__ = "Copyright (c) 2012 Kui Zeng"
__license__ = "Python"


import os 
import string 
import sys
import math

"""
Training the logistic regression model with gradient descent batch alogrithm
Problem source : http://openclassroom.stanford.edu/MainFolder/DocumentPage.php?course=MachineLearning&doc=exercises/ex4/ex4.html
"""

class LogisticRegression :
    def __init__ ( self ) :
        self.__X = []
        self.__Y = []    
        self.__theta = []
        self.__LEARNING_RATE = 7 
        self.__FEATURE_CNT = 1 + 2
        self.__load_training_data ()
        self.__SAMPLE_CNT = len ( self.__Y )
        self.__feature_scaling ()

        for idx in range ( 0, self.__FEATURE_CNT ) :
            self.__theta.append(0)

    def __load_training_data(self) :
        fp = open ( "./ex4x.dat", "r" ) 
        for line in fp.readlines() :
            (x1, x2) = line.strip('\r\n').split ( '\t' )
            self.__X.append ( [1, float(x1), float(x2)] )    
        fp.close()    

        fp = open ( "./ex4y.dat", "r" ) 
        for line in fp.readlines() :
            y = line.strip('\r\n\t')
            self.__Y.append ( float(y) )    
        fp.close()        

    def __feature_scaling(self) :
        max_value = []
        min_value = []
        for fidx in range ( 0, self.__FEATURE_CNT ) :
            max_value.append(0)
            min_value.append(100)

        for idx in range ( 0, self.__SAMPLE_CNT) :
            for fidx in range ( 1, self.__FEATURE_CNT ) :
                if max_value[fidx] < self.__X[idx][fidx] :
                    max_value[fidx] = self.__X[idx][fidx]
                if min_value[fidx] > self.__X[idx][fidx] :
                    min_value[fidx] = self.__X[idx][fidx]
        for idx in range ( 0, self.__SAMPLE_CNT) :
            x = self.__X[idx]
            for fidx in range ( 1, self.__FEATURE_CNT ) :
                self.__X[idx][fidx] = ( x[fidx] - min_value[fidx] ) / ( max_value[fidx] - min_value[fidx] )  


    def batch_learning_alogrithm (self) :
        last_loss = 0
        for  itr in range ( 1, 100000 ) :    
            self.__training () 
            loss = self.__get_loss ()
            sys.stdout.write ( "After %s iteratorion loss = %lf\n" % (itr, loss) )

            if math.fabs ( loss - last_loss)  <= 0.01 :
                break;
            last_loss = loss

        sys.stdout.write ( "The coef of the logistic model :\n")
        for idx in range ( 0, self.__FEATURE_CNT ) :
            sys.stdout.write ( "theta[%d] = %lf\n" % ( idx, self.__theta[idx]) )
            
    def __training (self) :

        weight = []
        for idx in range ( 0, self.__FEATURE_CNT ) :
            weight.append(0) 
    
        """calcaulate the loss for all samples"""
        for idx in range ( 0, self.__SAMPLE_CNT) :
            x = self.__X[idx]
            y = self.__Y[idx]
            h = self.__sigmoid( x ) 
            for fidx in range ( 0, self.__FEATURE_CNT ) :
                weight[fidx] +=  ( h - y ) * x[fidx]

        """update the weight all at once"""    
        for idx in range ( 0, self.__FEATURE_CNT ) :
            self.__theta[idx] -= self.__LEARNING_RATE * weight[idx] / self.__SAMPLE_CNT


        
    def __get_loss (self ) :
        loss = 0
        for idx in range ( 0, self.__SAMPLE_CNT) :
            x = self.__X[idx]
            y = self.__Y[idx]
            h = self.__sigmoid( x ) 
            loss += y * math.log (h) + ( 1 - y ) * math.log ( 1 - h )
        return loss
 
    def __sigmoid ( self, x ) :
        logit = 0
        for idx in range ( 0, self.__FEATURE_CNT):
            logit += self.__theta[idx] * x[idx]
        return 1.0 / ( 1.0 + math.exp ( -logit ) )

    def test ( self ) :
        wrong_ans = 0
        for idx in range ( 0, self.__SAMPLE_CNT) :
            x = self.__X[idx]
            y = self.__Y[idx]
            h = self.__sigmoid( x ) 
            check = 0
            if y > 0.5 and h< 0.5  :
                check = -1
            if y < 0.5 and h > 0.5 :
                check = -1
            sys.stdout.write ( "sample %d : ANS = %.2lf, TEST = %.2lf check = %d\n" % ( idx, y, h, check ))    
            wrong_ans -= check    
        print "wrong ans = %d" % wrong_ans
            
if __name__ == "__main__" :

    lr = LogisticRegression();
    lr.bath_learning_alogrithm ()
    lr.test()
