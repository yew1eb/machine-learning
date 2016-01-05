#!/usr/local/bin/python
# -*- coding : utf-8 -*-

__author__ = "zengkui111@gmail.com"
__version__ = "$Revision: 1.0 $"
__date__ = "$Mon 10 Dec 2012 11:04:16 PM CST $"
__fileName__ = "adaboost.py"
__license__ = "Python"

import os
import sys
import math

inf = float("inf")

def I(b, v1 = 1, v2 = 0):
    return v1 if b else v2

class Point:
    def __init__(self, x, y, label, weight ):
        self.x = x
        self.y = y
        self.label = label
        self.weight = weight 

class Learner:
    def __init__(self, vrng, fid ):
        self.vrng = vrng
        self.fid = fid
        fx = lambda d: I(vrng[0] < d.x < vrng[1], 1.0, -1.0)
        fy = lambda d: I(vrng[0] < d.y < vrng[1], 1.0, -1.0)
        self.predict = I( fid == 'x', fx, fy)
        self.weight = 0.0

class Adaboost:
    def __init__(self, input_data, max_iteration):
        self.read_data(input_data)

        weak_learners = []
        boundary = []
        for p in self.points :
            boundary.append(p.x)
        weak_learners = self.make_learners( boundary, 'x')

        boundary = []
        for p in self.points :
            boundary.append(p.y)
        weak_learners += self.make_learners( boundary, 'y')
        sys.stderr.write( "weak leaners %d\n" % len(weak_learners)) 
        self.learners = []
        for ite in xrange(max_iteration):
            #calculate the error for each learner
            info = []
            for learner in weak_learners:
                err = 0.0
                for p in self.points :
                    category = learner.predict(p)
                    if category != p.label: 
                        err += p.weight
                info.append((math.fabs(0.5 - err), err, learner))
         
            [terr, err, learner]= max(info)
            learner.weight = (math.log((1- err) / err))/2
            print "iteration", ite,
            print "[", learner.vrng,learner.fid, "]", learner.weight
            #update the weight of the learner
            self.learners.append(learner)
            #update the weight of the all samples 
            z = 0.0
            for p in self.points:
                category =  learner.predict(p)
                t = p.weight * math.exp( -1 * learner.weight * p.label * category)
                p.weight = t
                z += t
            for p in self.points:
                p.weight = p.weight / z
 
    def read_data(self, input_data):
        fp = open(input_data)
        self.points = []
        pn = fp.readline().strip('\r\n')
        pn = float(pn)
        weight = 1.0 / pn
        for line in fp.readlines():
            x,y,label = line.strip('\r\n') .split(',')
            self.points.append( Point(float(x), float(y), float(label), weight ))
        print len(self.points)
        fp.close()

    def make_learners(self, boundary, fid):
        boundary = [min(boundary) - 1] + sorted(boundary) + [max(boundary) + 1]
        learners = []
        for i in range(1, len(boundary)):
            if boundary[i] == boundary[i - 1]:
                continue
            l = (boundary[i] + boundary[i - 1]) / 2
            learners.append(Learner((l, inf), fid))
            learners.append(Learner((-inf, l), fid))
        return learners 

    def predict(self):
        print "(x, y)\tlabel\tpredict"
        for p in self.points:
            category = 0.0 
            for learner in self.learners:
                category += learner.predict(p) * learner.weight
            if category > 0 :
                print "(%.f, %.f)\t%+.f\t%+.f" % (p.x, p.y, p.label, 1)
            else :
                print "(%.f, %.f)\t%+.f\t%+.f" % (p.x, p.y, p.label, -1)


if __name__ == "__main__":

    adaboost = Adaboost("data.in", 20) 
    adaboost.predict()
