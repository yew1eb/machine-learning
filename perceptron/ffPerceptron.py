import random
import numpy as np

class Perceptron(object):
    '''
    simple feed forward Perceptron
    no bias yet, will add later
    '''
    def __init__(self, alpha, iterations):
        self.alpha = alpha
        self.iterations = iterations
        # initialize weights
        self.w = None

    def sigmoid(self, x):
        """
        Typical sigmoid function created from input vector x
        :param x: input vector
        :return: sigmoided vector
        """
        # typical sigmoid py line, seems to get errors with arrays
        return 1 / (1 + np.exp(-x))

    def response(self, X):
        '''
        Perceptron response
        :param X: input vector
        :return: Perceptron out
        '''
        y = self.sigmoid(np.dot(X, self.w))
        # in this super simple example Perceptron will return 1 for positive guess and -1 for neg
        if y >= 0.5:
            return 1
        else:
            return 0

    def updateWeight(self, X, error):
        '''
        update the vector of input weights
        :param X: input data
        :param error: prediction != true
        :return: updated weight vector
        '''
        self.w = self.w + self.alpha * np.dot(error, X)

    def train(self, X, y):
        '''
        trains Perceptron on vector data by looping each row and updating the weight vector
        :param X: input data
        :param y: correct y value
        :return: updated parameters
        '''
        num_examples, num_features = np.shape(X)

        # set up bias
        bias = np.ones(shape =(num_examples,1))
        X = np.hstack((bias, X))
        print(X[1])

        # initialize weight vector
        self.w = np.random.rand(num_features + 1)
        print('starting weights:', self.w)

        for i in range(self.iterations):
            for j in range(num_examples):
                prediction = self.response(X[j])
                if prediction != y[j]:
                    error = y[j] - prediction
                    self.updateWeight(X[j], error)