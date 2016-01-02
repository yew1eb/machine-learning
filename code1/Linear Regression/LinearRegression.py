import random
import numpy


class LinearRegression:

    # initialize

    def __init__(self, X, Y, alpha=0.0005, lam=0.1, printIter=True):

        x = numpy.array(X)
        m, n = x.shape

        # normalize data
        self.xMean = numpy.mean(x, axis=0)
        self.xStd = numpy.std(x, axis=0)
        x = (x - self.xMean) / self.xStd

        # add const column to X
        const = numpy.array([1] * m).reshape(m, 1)
        self.X = numpy.append(const, x, axis=1)

        self.Y = numpy.array(Y)
        self.alpha = alpha
        self.lam = lam
        self.theta = numpy.array([0.0] * (n + 1))

        self.printIter = printIter

    # caluclate cost
    def __costFunc(self):
        "calculate sum square error"
        m, n = self.X.shape
        pred = numpy.dot(self.X, self.theta)
        err = pred - self.Y
        cost = sum(err ** 2) / (2 * m) + self.lam * \
            sum(self.theta[1:] ** 2) / (2 * m)
        return(cost)

    # gradient descend
    def __gradientDescend(self, iter):
        """
        gradient descend:
        X: feature matrix
        Y: response
        theta: predict parameter
        alpha: learning rate
        lam: lambda, penality on theta
       """

        m, n = self.X.shape

        # print "m,n=" , m,n
        # print "theta", len(self.theta)

        for i in range(0, iter):
            theta_temp = self.theta
            # update theta[0]
            pred = numpy.dot(self.X, self.theta)
            err = pred - self.Y

            # print "grad" , self.alpha*(1.0/m)*sum(err*self.X[:,0])

            self.theta[0] = theta_temp[0] - self.alpha * \
                (1.0 / m) * sum(err * self.X[:, 0])
            for j in range(1, n):
                val = theta_temp[
                    j] - self.alpha * (1.0 / m) * (sum(err * self.X[:, j]) + self.lam * m * theta_temp[j])
                # print val
                self.theta[j] = val
            # calculate cost and print
            cost = self.__costFunc()

            if self.printIter:
                print "Iteration", i, "\tcost=", cost
                # print "theta", self.theta

    # simple name
    def run(self, iter, printIter=True):
        self.printIter = printIter
        self.__gradientDescend(iter)

    # prediction
    def predict(self, X):

        # add const column
        m, n = X.shape
        x = numpy.array(X)
        x = (x - self.xMean) / self.xStd
        const = numpy.array([1] * m).reshape(m, 1)
        X = numpy.append(const, x, axis=1)

        pred = numpy.dot(X, self.theta)
        return pred


def main():
    print "This is a simple linear regression test..."
    # generate feature X
    x = numpy.arange(0, 20).reshape(10, 2)

    # generate sample response
    y = numpy.arange(
        0, 10) + numpy.array([random.random() for r in range(0, 10)])

    lm_model = LinearRegression(x, y)
    lm_model.run(100)
    lm_model.predict(x)


if __name__ == "__main__":
    main()
