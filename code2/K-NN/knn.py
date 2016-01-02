#!/usr/local/bin/python
# -*- coding : utf-8 -*-

__author__ = "zengkui111@gmail.com"
__version__ = "$Revision: 1.0 $"
__date__ = "$Mon 05 Nov 2012 04:51:41 PM CST $"
__fileName__ = "knn.py"
__copyright__ = "Copyright (c) 2012 domob"
__license__ = "Python"

import os
import sys
import math
import argparse
import model_evaluate

"""
Description : 
    K-NN classification algorithm. 
    Data is represented by vsm and the features in vsm are the words selected by max info gain.
    The distance between samples is measured by cosine and Euclidean distance.
ChangeLog : 
"""
class Sample:
    def __init__(self):
        self.label = None
        self.vsm = {}
    def load_sample(self, word_list, label, word_dict ):
        self.label = label
        for w in word_list :
            if len(w) <= 3 :
                continue
            if w not in word_dict :
                continue
            inc(self.vsm, w, 1)
        total = 0
        for w in self.vsm:
            total += self.vsm[w] * self.vsm[w] 
        total  = math.sqrt(total)
        for w in self.vsm:
            self.vsm[w] = self.vsm[w] * 1.0 / total

def inc(mydict, key, value):
    if key not in mydict:
        mydict[key] = 0
    mydict[key] += value


        
class KNNClassifier:
    def __init__(self, K = None, dist_func = "c"):
        self.__word_dict = set()
        self.__training_samples = []
        self.__K = 11
        if K != None :
            self.__K = K 
            sys.stderr.write ( "K number = %d!\n" %  self.__K)
        else :
            sys.stderr.write ( "Default K = 11!\n")


        self.__dist_type = "cosine"
        if dist_func == "e" :
            self.__dist_type = "euclidean" 
            sys.stderr.write ( "Distance function is euclidean !\n")
        else :
            sys.stderr.write ( "Cosine distance is default !\n")
        

    def __entropy(self, num, den) :
        if  num == 0:
            return 0
        p = float(num)/float(den)
        return -p*math.log(p,2)

    def __get_word_dict(self, train_file):
        fp = open(train_file)
        total_positive = 0 
        total_negative = 0 
        df = {}
        total = 0 
        while True:
            line = fp.readline()
            if len(line) <= 0:
                break
            arr = line.strip('\r\n').split('\t')
            label = int(arr[0])
            if label > 0:
                total_positive += 1
            else:
                total_negative += 1
            word = set()
            for w in arr[1:]:
                if len(w) <= 3:
                    continue
                word.add(w)
            for w in word:
                if w not in df:
                    df[w] = { 1:0, -1:0}
                inc(df[w], label, 1)
            total += 1

        info = self.__entropy(total_positive, total) + self.__entropy(total_negative, total)
        word_ig = []
        for w in df:
            ig = info * 1000.0

            f = float(df[w][1] + df[w][-1]) 
            wi = f / total 
            ig -= 1000.0 * wi * (self.__entropy(df[w][1], f) + self.__entropy(df[w][-1], f))

            nf = total - f
            wi = float(nf)/total
            ig -= 1000.0 *wi * (self.__entropy(total_positive - df[w][1], nf) + self.__entropy(total_negative - df[w][-1], nf))
            word_ig.append((ig,w))
        word_ig.sort()
        word_ig.reverse()
        for ig,w in word_ig[:50]:
            self.__word_dict.add(w)


    def load_training_sample(self, train_file):
        self.__get_word_dict(train_file)
        fp = open(train_file)
        while True :
            line = fp.readline()
            if len(line) <= 0 :
                break
            arr = line.strip("\r\n").split('\t')
            label = int(arr[0])
            s = Sample()
            s.load_sample(arr[1:], label, self.__word_dict)
            self.__training_samples.append(s)
        fp.close()

    def __distance(self, s1, s2):
        if self.__dist_type == "euclidean" :
            return self.__euclidea_distance(s1, s2)
        else:
            return self.__cosine(s1, s2)

    def __euclidea_distance( self, s1, s2):
        dist = 0.0
        for w in self.__word_dict:
            x = 0.0
            y = 0.0
            if w in s1.vsm:
                x = s1.vsm[w]
            if w in s2.vsm:
                y = s2.vsm[w]
            dist += (x - y) * ( x - y)
        return math.sqrt(dist)

    def __cosine(self, s1, s2 ):
        num = 0.0
        den1 = 0.0
        den2 = 0.0
        for w in self.__word_dict:
            if w in s1.vsm and w in s2.vsm:
                num += s1.vsm[w] * s2.vsm[w]
            if w in s1.vsm:
                den1 += s1.vsm[w] * s1.vsm[w]
            if w in s2.vsm:
                den2 += s2.vsm[w] * s2.vsm[w]
        if den1 * den2 == 0 :
            return 0.0
        return num/(math.sqrt(den1) * math.sqrt(den2))

    def __classifier(self, test_sample):
        category = []
        for s in self.__training_samples:
            dist = self.__distance(s, test_sample)
            category.append((dist, s.label))
        category.sort()
        if self.__dist_type == "cosine" :
            category.reverse()
        positive = 0.0
        for d,l in category[:self.__K]:
            if l >= 0:
                positive += 1.0
        return positive/self.__K

    def predict(self, test_file, output_file):
        total = 0.0 
        wfp = open(output_file, "w")
        fp = open(test_file)
        positive = 0
        negative = 0
        true_positive = 0
        false_positive = 0
        true_negative = 0
        false_negative = 0
        sys.stderr.write( "start to predict ...\n" )
        me =  model_evaluate.ModelEvaluate()
        while True:
            line = fp.readline()
            if len(line) <= 0 :
                break
            arr = line.strip('\r\n').split('\t')
            label = int(arr[0])
            test_sample = Sample()
            test_sample.load_sample(arr[1:], label, self.__word_dict)
            p = self.__classifier(test_sample)
            wfp.write( "%d\t%f\n" % (label, p))
            total += 1
            if p > 0.5:
                p = 1
            else :
                p = -1
            me.add( int(label), p )
        me.report()
        fp.close()
        wfp.close()
                 
    

if __name__ == "__main__" : 

    parser = argparse.ArgumentParser( description = "KNN classifer" )
    parser.add_argument( "-d", "--dist_function", help = "distance function [e|c]")
    parser.add_argument( "-s", "--train_file", help = "sample file")
    parser.add_argument( "-t", "--test_file", help = "test file")
    parser.add_argument( "-o", "--output_file", help = "output file")
    parser.add_argument( "-k", "--k_nn", help = "K number")
    args = parser.parse_args()

    train_file = args.train_file 
    test_file = args.test_file
    output_file = args.output_file
    dist_func = args.dist_function

    if args.k_nn :
        k_number = int(args.k_nn)
    if not train_file or not os.path.exists(train_file):
        parser.print_help()
        sys.exit(1)
    if not test_file or not os.path.exists(test_file):
        parser.print_help()
        sys.exit(1)
   
    knn = KNNClassifier(k_number, dist_func)
    knn.load_training_sample(train_file)
    knn.predict(test_file, output_file)
