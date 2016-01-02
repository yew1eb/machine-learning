import matplotlib.pyplot as plt
import numpy as np
import random

from sklearn import datasets, linear_model

def scale(X, axis=0):
    new = X - np.mean(X, axis=0)
    return new / np.std(new, axis=0)

class LinearRegression(object):
    def __init__(self, eta=0.001, n_iter=50, fit_alg='sgd'):
        self.eta = eta
        self.n_iter = n_iter
        self.fit_alg = self.fit_sgd if fit_alg=='sgd' else self.fit_batch

    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)
        self.w_ = np.ones(X.shape[1])

        for _ in range(self.n_iter):
            self.fit_alg(X, y)

        return self

    def fit_batch(self, X, y):
        output = X.dot(self.w_)
        errors = y - output
        self.w_ += self.eta * X.T.dot(errors)
        # print(sum(errors**2) / 2.0)

    def fit_sgd(self, X, y):
        X, y = self._shuffle(X, y)
        for x, target in zip(X, y):
            output = x.dot(self.w_)
            errors = target - output
            self.w_ += self.eta * x.T.dot(errors)

    def _shuffle(self, X, y):
        r = np.random.permutation(len(y))
        return X[r], y[r]

    def predict(self, X):
        return np.insert(X, 0, 1, axis=1).dot(self.w_)

    def score(self, X, y):
        return 1 - sum((self.predict(X) - y)**2) / sum((y - np.mean(y))**2)


diabetes = datasets.load_diabetes()
# Use only one feature
diabetes_X = diabetes.data[:, np.newaxis, 2]
diabetes_X = scale(diabetes_X)
diabetes_y = scale(diabetes.target)

diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]
# diabetes_y_train = diabetes.target[:-20]
# diabetes_y_test = diabetes.target[-20:]
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

# regr = linear_model.LinearRegression()
regr = LinearRegression(n_iter=50, fit_alg='sgd')
regr.fit(diabetes_X_train, diabetes_y_train)

# regr.fit(np.array([[0, 0], [1, 1], [2, 2]]), np.array([0, 1, 2]))
# print(regr.predict(np.array([[3, 3]])))
# print(regr.predict(diabetes_X_test))

# print('Coefficients: \n', regr.coef_)
# print("Residual sum of squares: %.2f"
#       % np.mean((regr.predict(diabetes_X_test) - diabetes_y_test) ** 2))
print('Variance score: %.2f' % regr.score(diabetes_X_test, diabetes_y_test))

# plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
# plt.plot(diabetes_X_test, regr.predict(diabetes_X_test), color='blue', linewidth=3)
# plt.xticks(())
# plt.yticks(())
# plt.show()







