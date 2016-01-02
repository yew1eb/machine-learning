#!/usr/local/bin/python
# -*- coding : utf-8 -*-

import sys
import os
import math
import argparse

"""
Author : zengkui111@gmail.com
Created Time : Sun 03 Nov 2012 10:12:47 PM CST
FileName : decision_tree.py
Description : text classifier 
ChangeLog : 
"""

did2label = {}
wid2word = {}
didwordlist = {} 
widdoclist = {}

def load_train_data( file_path ):

    fp = open(file_path)
    did = 0
    word_idx = {}
    wid = 0
    doc_list = set()
    while True :
        line = fp.readline()
        if len(line) <= 0 :
            break
        arr = line.strip('\r\n').split('\t')
        label = int(arr[0])
        did2label[did] = label
        didwordlist[did] = set()
        for w in arr[1:]:
            if len(w) <= 3 :
                  continue
            if w not in word_idx:
                word_idx[w] = wid                   
                wid2word[wid] = w
                widdoclist[wid] = set()
                wid += 1
            widdoclist[word_idx[w]].add(did)
            didwordlist[did].add(word_idx[w])
        doc_list.add(did)
        did += 1
    return doc_list
    
def entropy( num, den ):
    if num == 0 :
        return 0
    p = float(num)/float(den)   
    return -p*math.log(p,2)


class DecisionTree :
    def __init__(self) :
        self.word = None
        self.doc_count = 0
        self.positive = 0
        self.negative = 0
        self.child = {}

    def predict(self, word_list ):
        if len(self.child) == 0 :
                return float(self.positive)/(self.positive+self.negative)
        if self.word in word_list :
            return self.child["left"].predict(word_list)
        else :
            return self.child["right"].predict(word_list)

    def visualize(self, d) :
        "visualize the tree"
        for i in range (0, d) :
            print "-",
        print "(%s,%d,%d)" % ( self.word,self.positive, self.negative)
        if len(self.child) != 0 :
            self.child["left"].visualize(d + 1)
            self.child["right"].visualize(d + 1)
         
    def build_dt(self, doc_list ) :
        self.doc_count = len(doc_list)
        for did in doc_list :
            if did2label[did] > 0 :
                self.positive += 1
            else :
                self.negative += 1

        if self.doc_count <= 10 or self.positive * self.negative == 0 : 
            return True            
        wid = info_gain( doc_list )
        if wid == -1 : 
            return True        
        self.word = wid2word[wid]
        left_list = set() 
        right_list = set() 
        for did in doc_list :
            if did in widdoclist[wid] :
                left_list.add(did)
            else :
                right_list.add(did)

        self.child["left"] =  DecisionTree()
        self.child["right"] =  DecisionTree()
        self.child["left"].build_dt( left_list )
        self.child["right"].build_dt(right_list )

def info_gain(doc_list):
    collect_word = set()
    total_positive = 0
    total_negative = 0
    for did in doc_list :
        for wid in didwordlist[did] :
            collect_word.add(wid)
        if did2label[did] > 0 :
            total_positive += 1
        else :
            total_negative += 1
    total = len(doc_list)
    info = entropy( total_positive, total )
    info += entropy( total_negative, total )
    ig = []
    for wid in collect_word :
        positive = 0
        negative = 0
        for did in widdoclist[wid]:
            if did not in doc_list :
                continue
            if did2label[did] > 0 :
                positive += 1
            else :
                negative += 1
        df = negative + positive 
        a = info
        b = entropy( positive, df )     
        b += entropy( negative, df )     
        a -= b * df / total

        b = entropy( total_positive - positive, total - df)     
        b += entropy( total_negative - negative, total - df )     
        a -= b * ( total - df ) / total
        a = a * 100000.0
        ig.append( (a, wid))
    ig.sort()
    ig.reverse()
    for i,wid in ig :
        left = 0
        right = 0
        for did in doc_list :
            if did in widdoclist[wid] :
                left += 1
            else :
                right += 1
        if left >= 5 and right >= 5 :
            return wid
    return -1 




if __name__ == "__main__" :

    parser = argparse.ArgumentParser( description = "Decision Tree training and testing" )
    parser.add_argument( "-i", "--train_data", help = "training data")
    parser.add_argument( "-t", "--test_data", help = "testing data")
    args = parser.parse_args()
    
    train_file = args.train_data
    test_file  = args.test_data
    if not train_file or not os.path.exists(train_file) :
        parser.print_help()
        sys.exit()
    if not test_file or not os.path.exists(test_file) :
        parser.print_help()
        sys.exit()
    
    doc_list = load_train_data( train_file )

    dt = DecisionTree()
    dt.build_dt(doc_list)
    #dt.visualize(0)

    fp = open(test_file)
    true_positive = 0
    false_positive = 0
    positive = 0
    true_negative = 0
    false_negative = 0
    negative = 0
    total = 0
    while True :
        line = fp.readline()
        if len( line ) <= 0 :
            break
        arr = line.strip('\r\n').split('\t')
        label = int(arr[0])
        word_list = set() 
        for w in arr[1:] :
            if len(w) <= 3 :
                continue
            word_list.add( w )
        p = dt.predict(word_list)
        print label, p
        if label == 1 :
            positive += 1
        else :
            negative += 1
        if p >= 0.5 :
            if label == 1 : 
                true_positive += 1
            else :
                false_positive += 1
        else :
            negative += 1
            if label == -1 :
                true_negative += 1
            else :
                false_negative += 1
        total += 1
    print "Positive recall :%f" % (true_positive*100.0/(positive))
    print "Positive precision :%f" % (true_positive*100.0/(true_positive+false_positive))
    print "Accuary : %f" % ( (true_positive + true_negative)*100.0/total)
             


