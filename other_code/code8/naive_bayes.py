#!/usr/bin/env python
# -*- coding: utf-8 -*-

import multivariate_normal


def main():
    # load sample data
    X, X_labels = multivariate_normal.load_data_with_label()
    n_labels = max(X_labels)+1
    n_features = len(X[0])


if __name__ == '__main__':
    main()
