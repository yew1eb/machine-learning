#-*- coding: utf-8 -*-

import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer
from MLP import MLP


def main():
    digits = load_digits()
    X = digits.data
    y = digits.target
    X -= X.min()
    X /= X.max()

    mlp = MLP(64, 100, 10)
    mlp.print_configuration()

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    labels_train = LabelBinarizer().fit_transform(y_train)
    labels_test = LabelBinarizer().fit_transform(y_test)

    mlp.fit(X_train, labels_train)
    predictions = []
    for i in range(X_test.shape[0]):
        o = mlp.predict(X_test[i])
        predictions.append(np.argmax(o))
    print confusion_matrix(y_test, predictions)
    print classification_report(y_test, predictions)

if __name__ == '__main__':
    main()
