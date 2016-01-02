import math
import numpy as np


class Logit(object):
    """
    logistic regression using gradient descent
    """
    def __init__(self, learning_rate = 0.01, iterations = 100, verbose = True, 
                tolerance = 0, l2 = 0, intercept = True):
        """
        :param learning_rate: learning rate constant
        :param iterations: how many epochs
        :param tolerance: the error value in which to stop training
        :param intercept: whether to fit an intercept
        :param verbose: whether to spit out error rates while training
        :param l2: L2 regularization term
        """
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.tolerance = tolerance
        self.intercept = intercept
        self.verbose = verbose
        self.l2 = l2
        self.theta = None
        self.mean = []
        self.std = []

    def sigmoid(self, x):
        """
        Typical sigmoid function created from input vector x

        :param x: input vector
        :return: sigmoided vector
        """
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        """
        Gradient descent, loops over theta and updates to
        take steps in direction of steepest decrease of J.
        :input x: must be numpy array
        :input y: must be numpy vector of 0 and 1
        :return: value of theta that minimizes J(theta) and J_history
        """
        if self.intercept:
            intercept = np.ones((np.shape(X)[0],1))
            X = np.concatenate((intercept, X), 1)
        
        num_examples, num_features = np.shape(X)

        # initialize theta to 1
        self.theta = np.ones(num_features)

        for i in range(self.iterations):
            # make predictions
            predicted = self.sigmoid(np.dot(X, self.theta.T))
            # update theta with gradient descent
            self.theta = (self.theta * (1 - (self.learning_rate * self.l2))) - self.learning_rate * np.dot((predicted - y).T, X)
            # sum of squares cost
            error = predicted - y
            cost = np.sum(error**2) / (2 * num_examples)

            if i % (self.iterations/10) == 0 and self.verbose == True:
                print 'iteration:', i
                print 'theta:', self.theta
                print 'cost:', cost

            if cost < self.tolerance:
                return self.theta
                break

        return self.theta

    def predict(self, X, labels):
        """
        Make linear prediction based on cost and gradient descent

        :param X: new data to make predictions on
        :param labels: boolean
        :return: return prediction
        """
        if self.intercept:
            intercept = np.ones((np.shape(X)[0],1))
            X = np.concatenate((intercept, X), 1)
            
        num_examples, num_features = np.shape(X)
        prediction = []
        for sample in range(num_examples):
            yhat = 0
            for value in range(num_features):
                yhat += X[sample, value] * self.theta[value]
            
            pred = self.sigmoid(yhat)
            
            if labels:
                if pred > 0.5:
                    prediction.append(int(1))
                else:
                    prediction.append(int(0))
            else:
                prediction.append(yhat)   
                
        return prediction
        

def demo():
    from sklearn.cross_validation import train_test_split
    from sklearn.metrics import confusion_matrix, classification_report 
    # initialize linear regression parameters
    max_iterations = 50000
    learning_rate = 0.0001
    l2 = 1.0

    lgit = Logit(learning_rate = learning_rate, iterations = max_iterations, 
                verbose = True, tolerance = 0.001, l2 = l2)

    data = np.loadtxt('Data/ionosphere.csv', delimiter = ',')
    X = data[:, 1:]
    y = data[:, 0]
    
    # scale data
    max = np.amax(X)
    X /= max
    
    prediction = []
    correct = []
    for i in range(0,10):
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

        # fit the reg
        lgit.fit(X = X_train, y = y_train)
    
        # make a predictions
        prediction.append(lgit.predict(X_test, labels = True))
        correct.append(y_test.tolist())
    
    print classification_report(np.array(correct), np.array(prediction))
    
if __name__ == '__main__':
    demo()