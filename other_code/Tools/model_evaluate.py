#!/usr/local/bin/python
# -*- coding : utf-8 -*-

__author__ = "zengkui111@gmail.com"
__version__ = "$Revision: 1.0 $"
__date__ = "$Thu 08 Nov 2012 10:06:16 PM CST $"
__fileName__ = "model_evaluate.py"
__copyright__ = "Copyright (c) 2012 domob"
__license__ = "Python"

import os
import sys
import math
import argparse

class ModelEvaluate:
    def  __init__(self):
        self.positive = 0
        self.negative = 0
        self.true_positive = 0
        self.true_negative = 0
        self.false_positive = 0
        self.false_negative = 0
        self.total = 0
    def add(self, label, p):
        """p is the predicted label [+1|-1]"""
        if label == 1 :
            self.positive += 1
        else :
            self.negative += 1
        if p > 0:
            if label == 1 : 
                self.true_positive += 1 
            else : 
                self.false_positive += 1 
        else :
            if label == -1 :
                self.true_negative += 1
            else :
                self.false_negative += 1
        self.total += 1

    def report(self):
        recall = self.true_positive * 100.0 / self.positive
        sys.stderr.write ( "Positive recall :%f\n" % recall) 
        precision  = self.true_positive * 100.0 / (self.true_positive + self.false_positive)
        sys.stderr.write ( "Positive precision :%f\n" % precision)
        accuarcy  = (self.true_positive + self.true_negative) * 100.0 / self.total
        sys.stderr.write ( "Accuarcy : %f\n" %  accuarcy)
             


