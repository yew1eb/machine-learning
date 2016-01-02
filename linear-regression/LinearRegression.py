import numpy as np

class LinReg(object):
    """
    multivariate linear regression using gradient descent
    """
    def __init__(self, learning_rate = 0.01, iterations = 50, verbose = True, l2 = 0, 
                tolerance = 0, intercept = True):
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

    def fit(self, X, y):
        """
        Gradient descent, loops over theta and updates to
        take steps in direction of steepest decrease of J.
        :return: value of theta that minimizes J(theta) and J_history
        """
        if self.intercept:
            intercept = np.ones((np.shape(X)[0],1))
            X = np.concatenate((intercept, X), 1)
            
        num_examples, num_features = np.shape(X)

        # initialize theta to 1
        self.theta = np.ones(num_features)

        for i in range(self.iterations):
            # make prediction
            predicted = np.dot(X, self.theta.T)
            # update theta with gradient descent
            self.theta = (self.theta * (1 - (self.learning_rate * self.l2))) - self.learning_rate / num_examples * np.dot((predicted - y).T, X)
            # sum of squares cost
            error = predicted - y
            cost = np.sum(error**2) / (2 * num_examples)
            
            if i % 10 == 0 and self.verbose == True:
                print('iteration:', i)
                print('theta:', self.theta)
                print('cost:', cost)
                
            if cost < self.tolerance:
                return self.theta
                break

        return self.theta

    def predict(self, X):
        """
        Make linear prediction based on cost and gradient descent
        :param X: new data to make predictions on
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
            prediction.append(yhat)
                
        return prediction

def demo():
    # initialize linear regression parameters
    iterations = 2000
    learning_rate = 0.1
    l2 = 0.0001

    linearReg = LinReg(learning_rate = learning_rate, iterations = iterations, verbose = 1, l2 = l2)

    data = np.genfromtxt('Data/blood_pressure.csv', delimiter = ',', skip_header = 1)
    X = data[:, 1:]
    y = data[:, 0]
    
    # scale data
    max = np.amax(X)
    X /= max
    print(X)
    print(y)

    # fit the linear reg
    linearReg.fit(X = X, y = y)

    # load testing dataset
    test = np.genfromtxt('Data/blood_pressure.csv', delimiter = ',', skip_header = 1)
    X_test = test[:, 1:]
    y_test = test[:, 0]
    
    max = np.amax(X_test)
    X_test /= max
    print(X_test)

    predictions = np.array(linearReg.predict(X_test))

    print('correct: ', y_test)
    print('prediction: ', predictions)

if __name__ == '__main__':
    demo()