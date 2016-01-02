#-*- coding: utf-8 -*-

import numpy as np
from MLP import MLP


def main():
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    targets = np.array([0, 1, 1, 0])

    # initialize
    mlp = MLP(n_input_units=2, n_hidden_units=3, n_output_units=1)
    mlp.print_configuration()

    # training
    mlp.fit(inputs, targets)
    print '--- training ---'
    print 'first layer weight: '
    print mlp.v
    print 'second layer weight: '
    print mlp.w

    # predict
    print '--- predict ---'
    for i in [[0, 0], [0, 1], [1, 0], [1, 1]]:
        print i, mlp.predict(i)

if __name__ == '__main__':
    main()
