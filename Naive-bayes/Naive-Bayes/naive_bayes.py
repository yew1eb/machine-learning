#!/usr/local/bin/python
# -*- coding : utf-8 -*-

__author__ = "zengkui111@gmail.com"
__version__ = "$Revision: 1.0 $"
__date__ = "$Mon 05 Nov 2012 04:51:41 PM CST $"
__fileName__ = "Naive-Bayes.py"
__copyright__ = "Copyright (c) 2012 zengkui"
__license__ = "Python"

import os
import sys
import math
import argparse

import model_evaluate

"""
Description : 
    Naive Bayes classification algorithm.
    Bayes formular : P(C|w) = P(C,w) / P(w) = P(w|C) * P(C) / P(w)
    C : the category of articel.
    w : the words in articel
    Suppose that the probability of occurrence of a word in an article is independent
    We can classify the article by the following formular:
    P(C_i|w_1,w_2...w_n) = P(w_1,w_2...w_n|C_i) * P(C_i) / P(w_1,w_2...w_n)
    = P(w_1|C_i) * P(w_2|C_i)...P(w_n|C_i) * P(C_i) / (P(w_1) * P(w_2) ...P(w_n))
ChangeLog : 
"""

def inc(myhash, key, value):
    if key not in myhash:
        myhash[key] = 0
    myhash[key] += value

class NaiveBayes:
    def __init__(self):
        self.model = {}
        self.word_dict = set() 
        self.category_prob = {}

    def get_articel_words(self, text):
        arr = text.strip('\r\n').split('\t')
        label = arr[0]
        word = set()
        for w in arr[1:]:
            if w not in self.word_dict:
                continue
            word.add(w) 
        return (label, word)

    def train_model(self, path, dict_file):
        fp = open(dict_file)
        for line in fp.readlines():
            word = line.strip('\r\n')
            self.word_dict.add(word)
        fp.close()

        df = {}
        category = {}
        fp = open( path )
        while True:
            line = fp.readline()
            if len(line) <= 0 :
                break
            label,word = self.get_articel_words(line)
            if label not in category :
                category[label] = 0
            category[label] += 1
            for w in word:
                if w not in df:
                    df[w] = {}
                inc ( df[w], label, 1 )
                inc ( df[w], "total", 1 )
        fp.close()
        
        total = 0
        for l in category:
            total += category[l] 
        for w in df:
            if df[w]["total"] < total/10 :
                continue
            self.model[w] = {}
            line = w
            for l in category:
                if l not in df[w]:
                    self.model[w][l] =  0.00001 
                    line += "\t%d,%d,%.3lf" % ( 0, category[l], self.model[w][l] )
                else :
                    self.model[w][l] = df[w][l] * 1.0 / category[l]
                    line += "\t%d,%d,%.3lf" % ( df[w][l], category[l], self.model[w][l] )

        for l in category:
            self.category_prob[l] = category[l] * 1.0 / total 
    
    def predict(self, articel_words):

        cate = {}
        for l in self.category_prob:
            cate[l] = self.category_prob[l] 
        for w in articel_words: 
            if w not in self.model:
                continue
            for l in self.category_prob:
                if l in self.model[w] :
                    p = self.model[w][l] 
                    cate[l] = cate[l] * p 
        label = None
        max_prob = -1
        for l in cate :
            if max_prob < cate[l] :
                max_prob = cate[l]
                label = l
        if max_prob == 1.0 :
            return (-1,1)
        return (label, max_prob)

if __name__ == "__main__" : 

    parser = argparse.ArgumentParser( description = "Naive Bayss classifer" )
    parser.add_argument( "-s", "--train_file", help = "sample file")
    parser.add_argument( "-d", "--dict_file", help = "word dict file")
    parser.add_argument( "-t", "--test_file", help = "test file")
    parser.add_argument( "-o", "--output_file", help = "output file")
    args = parser.parse_args()

    train_file = args.train_file 
    test_file = args.test_file
    dict_file = args.dict_file
    output_file = args.output_file
    if not train_file or not os.path.exists(train_file):
        parser.print_help()
        sys.exit(1)
    if not test_file or not os.path.exists(test_file):
        parser.print_help()
        sys.exit(1)
    if not dict_file or not os.path.exists(dict_file):
        parser.print_help()
        sys.exit(1)
    
    me = model_evaluate.ModelEvaluate()
    naive_bayes = NaiveBayes()
    naive_bayes.train_model( train_file, dict_file)
    fp = open(test_file)
    while True:
        line = fp.readline()
        if len(line) <= 0 :
            break
        label, words = naive_bayes.get_articel_words(line)
        l,p = naive_bayes.predict(words)
        me.add(int(label), int(l))
    me.report()


