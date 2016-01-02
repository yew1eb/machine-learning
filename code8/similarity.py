#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Reference: http://dataaspirant.com/2015/04/11/five-most-popular-similarity-measures-implementation-in-python/
"""

import os
import numpy as np
from math import sqrt
from decimal import Decimal


def euclid_dist(x, y):
    return sqrt(sum(
        [(x[n] - y[n]) ** 2 for n in range(len(x))]
    ))


def manhattan_dist(x, y):
    return sum(
        [abs(xx - yy) for xx, yy in zip(x, y)]
    )


def nth_root(value, n_root):
    root_value = 1/float(n_root)
    return round(Decimal(value) ** Decimal(root_value), 3)


def minkowski_dist(x, y, p_value):
    return nth_root(sum(pow(abs(a-b), p_value) for a, b in zip(x, y)),
                    p_value)


def square_rooted(x):
    return round(sqrt(sum([a * a for a in x])), 3)


def cosine_similarity(x, y):
    numerator = sum(a * b for a, b in zip(x, y))
    denominator = square_rooted(x) * square_rooted(y)
    return round(numerator / float(denominator), 3)


def jaccard_similarity(x, y):
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))
    return intersection_cardinality / float(union_cardinality)


def main():
    print euclid_dist([0, 3, 4, 5], [7, 6, 3, -1])
    print manhattan_dist([10, 20, 10], [10, 20, 20])
    print minkowski_dist([0, 3, 4, 5], [7, 6, 3, -1], 3)
    print cosine_similarity([3, 45, 7, 2], [2, 54, 13, 15])
    print jaccard_similarity([0, 1, 2, 5, 6], [0, 2, 3, 5, 7, 9])


if __name__ == '__main__':
    main()
