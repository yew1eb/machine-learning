#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


def main():
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(digits.data,
                                                        digits.target)
    estimator = SVC(C=1.0, kernel='rbf', gamma=0.01)
    clf = OneVsRestClassifier(estimator)

    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, pred)
    print cm
    print accuracy_score(y_test, pred)

if __name__ == '__main__':
    main()
