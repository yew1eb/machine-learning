#!/usr/local/bin/python
# -*- coding : utf-8 -*-

__author__ = "zengkui111@gmail.com"
__version__ = "$Revision: 1.0 $"
__date__ = "$Sat 17 Nov 2012 02:47:03 PM CST $"
__fileName__ = "k.py"
__copyright__ = "Copyright (c) 2012 zengkui"
__license__ = "Python"

import sys 
import os 
import sys 
import math 
import argparse

"""
Algorithm description : 
    In data mining, k-means aims to partition N samples to K clusters.
How to judge two center is the same?
    the cosine distance between two centers is less than 0.0001
ChangeLog : 
"""



class KMeans:
    def __init__(self, dict_file):
        self.__word_dict = set()
        self.__load_word_dict(dict_file)
        self.__cluster_center = {} 

    def __load_word_dict(self, dict_file):
        fp = open(dict_file)
        while True:
            line = fp.readline()
            if len(line) <= 0 :
                break
            word = line.strip('\r\n\t ')
            self.__word_dict.add(word)
        fp.close()

    def __inc(self, d, k, v):
        if k not in d :
            d[k] = 0.0
        d[k] += v

    def __normalized(self, v):
        den = 0.0
        for k in v:
            den += v[k] * v[k]
        den = math.sqrt(den)
        for k in v:
            v[k] = v[k] / den
        return v

    def __get_vsm(self, words):
        v = {}
        for w in words:
            if w in self.__word_dict:
                self.__inc(v, w, 1.0)
        v = self.__normalized(v)
        return v

    def init_center(self, center_file):
        fp = open(center_file)
        while True:
            line = fp.readline()
            if len(line) <= 0:
                break
            words = line.strip('\r\n').split('\t')
            self.__cluster_center[words[0]] = self.__get_vsm(words[1:])
        fp.close()

            
    def __cosine(self, vsm_a, vsm_b):
        num = 0.0
        den1 = 0.0
        den2 = 0.0
        for w in self.__word_dict:
            a = 0.0
            b = 0.0
            if w  in vsm_a:
                a = vsm_a[w]
            if w in vsm_b:
                b = vsm_b[w]
            den1 += a * a 
            den2 += b * b 
            num += a * b 
        if den1 * den2 == 0 :
            return 0
        return num /(math.sqrt(den1) * math.sqrt(den2))

    def __euclidean(self, vsm_a, vsm_b):
        dist = 0.0
        for w in self.__word_dict:
            a = 0.0
            b = 0.0
            if w  in vsm_a:
                a = vsm_a[w]
            if w in vsm_b:
                b = vsm_b[w]
            dist += (a - b) * (a - b)
        return math.sqrt(dist) 

    def __distance(self, vsm_a, vsm_b, dtype = None ):
        if dtype == "euclidean" :
            dist = self.__euclidean( vsm_a, vsm_b )
        else :
            dist = 1 - self.__cosine( vsm_a, vsm_b )
        return dist

    def __assignment(self, article_vsm ):
        category = {}
        for aid in article_vsm:
            min_dist =   999999 
            label = None
            for l in self.__cluster_center:
                dist = self.__distance( article_vsm[aid], self.__cluster_center[l])
                if dist <  min_dist:
                    min_dist = dist 
                    label = l
            category[aid] = label
        return category

    def __adjust_center(self, category, article_vsm):
        center = {}
        counter = {}
        for aid in article_vsm:
            c = category[aid]
            if c not in  center:
                center[c] = {} 
            for w in article_vsm[aid]:
                self.__inc( center[c], w, article_vsm[aid][w])
            self.__inc(counter, c, 1.0) 
        for c in center:
            for w in center[c]:
                center[c][w] = float(center[c][w]) / float(counter[c])
        return center

    def __center_cmp(self, center):
        ret = False
        sys.stderr.write ("Center Diff :\n")
        for c in self.__cluster_center:
            dist = self.__distance( center[c], self.__cluster_center[c] )
            sys.stderr.write( "%s : %f\n" % ( c, dist))
            if dist >= 0.0001:
                ret = True 
        return ret 

    def cluster(self, data_file, output_file):
        fp = open(data_file)
        article_vsm = {} 
        article_id = 1 
        article_label = {}
        while True:
            line = fp.readline()
            if len(line) <= 0 :
                break
            words = line.strip('\r\n').split('\t')
            vsm = self.__get_vsm(words[1:])
            article_vsm[article_id] = vsm
            article_label[article_id] = words[0]
            article_id += 1
        fp.close()
        
        any_change = True
        itr = 1 
        while any_change : 
            sys.stderr.write( "iter : %d ...\n" % itr)
            category = self.__assignment( article_vsm ) 
            cluster_center = self.__adjust_center(category, article_vsm)
            any_change  = self.__center_cmp( cluster_center )
            self.__cluster_center = cluster_center
            itr += 1
        fp = open(output_file, "w")
        for aid in category:
            fp.write( "%s\t%s\n" % ( article_label[aid], category[aid]))
        fp.close()


if __name__ == "__main__" :


    parser = argparse.ArgumentParser( description = "k-means cluster" )
    parser.add_argument( "-w", "--word_dict", help = "word dict file")
    parser.add_argument( "-c", "--center_file", help = "center_file")
    parser.add_argument( "-i", "--data_file", help = "input data file")
    parser.add_argument( "-o", "--output_file", help = "output data file")
    args = parser.parse_args()
    
    word_dict = args.word_dict
    center_file = args.center_file
    data_file = args.data_file
    output_file = args.output_file
    if not word_dict or not os.path.exists(word_dict):
        sys.stderr.write( "word dict file may be not exists !!!\n")
        parser.print_help()
        sys.exit(1)
    if not center_file or not os.path.exists(center_file):
        sys.stderr.write( "center file may be not exists !!!\n")
        parser.print_help()
        sys.exit(1)
    if not data_file or not os.path.exists(data_file):
        sys.stderr.write( "input data file may be not exists !!!\n")
        parser.print_help()
        sys.exit(1)

    km  = KMeans(word_dict)
    km.init_center(center_file)
    km.cluster(data_file, output_file)

    

