#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
import multivariate_normal
import numpy as np


def main():
    X, X_labels = multivariate_normal.load_data_with_label()
    X_train, X_test, X_labels_train, X_labels_test = train_test_split(X,
                                                                      X_labels)
    clf = GaussianNB()
    clf.fit(X_train, X_labels_train)
    pred = clf.predict(X_test)
    X_labels_uniq = map(np.str, np.unique(X_labels))
    print classification_report(X_labels_test, pred,
                                target_names=X_labels_uniq)


if __name__ == '__main__':
    main()









