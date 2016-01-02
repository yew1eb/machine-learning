#-*- coding: utf-8 -*-

import numpy as np


class MLP(object):
    """
    3 Layered Perceptron
    """
    def __init__(self, n_input_units, n_hidden_units, n_output_units):
        self.nin = n_input_units
        self.nhid = n_hidden_units
        self.nout = n_output_units

        self.v = np.random.uniform(-1.0, 1.0, (self.nhid, self.nin+1))
        self.w = np.random.uniform(-1.0, 1.0, (self.nout, self.nhid+1))

    def fit(self, inputs, targets, learning_rate=0.2, epochs=10000):
        inputs = self.__add_bias(inputs, axis=1)
        targets = np.array(targets)

        for loop_cnt in xrange(epochs):
            # randomise the order of the inputs
            p = np.random.randint(inputs.shape[0])
            xp = inputs[p]
            bkp = targets[p]

            # forward phase
            gjp = self.__sigmoid(np.dot(self.v, xp))
            gjp = self.__add_bias(gjp)
            gkp = self.__sigmoid(np.dot(self.w, gjp))

            # backward phase(back prop)
            eps2 = self.__sigmoid_deriv(gkp) * (gkp - bkp)
            eps = self.__sigmoid_deriv(gjp) * np.dot(self.w.T, eps2)

            gjp = np.atleast_2d(gjp)
            eps2 = np.atleast_2d(eps2)
            self.w = self.w - learning_rate * np.dot(eps2.T, gjp)

            xp = np.atleast_2d(xp)
            eps = np.atleast_2d(eps)
            self.v = self.v - learning_rate * np.dot(eps.T, xp)[1:, :]

    def predict(self, x):
        x = np.array(x)
        x = self.__add_bias(x)
        hid = self.__sigmoid(np.dot(self.v, x))
        hid = self.__add_bias(hid)
        y = self.__sigmoid(np.dot(self.w, hid))
        return y

    def __add_bias(self, x, axis=None):
        return np.insert(x, 0, 1, axis=axis)

    def __sigmoid(self, u):
        """
        Sigmoid function(Activation function)
        """
        return (1.0 / (1.0 + np.exp(-u)))

    def __sigmoid_deriv(self, u):
        return (u * (1 - u))

    def print_configuration(self):
        print '--- NN configuration ---'
        print 'Num of input layer units: %d' % self.nin
        print 'Num of hidden layer units: %d' % self.nhid
        print 'Num of output layer units: %d' % self.nout
        print 'Shape of first layer weight(v):', self.v.shape
        print 'Shape of second layer weight(w):', self.w.shape
