import numpy as np

TRAIN_FILE = ''
TEST_FILE = ''


def load_file(file_name):
    with open(file_name, 'rb') as f:
        lines = f.readlines()
        lines = [l.strip('\n').split(' ') for l in lines]
        lines = [filter(lambda x: len(x) > 0, l) for l in lines]
        lines = [[float(i) for i in line] for line in lines]
        y = np.array([l[-1] for l in lines])
        X = np.array([l[:-1] for l in lines])
    return X, y


def euclidean_distance(x_1, x_2):
    """
    euclidean_distance. can ignore sqrt in the end in this case.
    """
    if x_1.shape != x_2.shape:
        raise ValueError("shape mismatch")
    diff = x_1 - x_2
    distance = np.sqrt(sum(np.power(diff, 2)))
    return distance


class KNN(object):

    """
    k nearest neighbor
    """

    def __init__(self, k, distance_func):
        self.k = k
        self.cal_dist = distance_func

    def fit(self, X, y):
        """
        lazy fit
        """
        self.train_X = X
        self.train_y = y
        return self

    def _single_predict(self, x):
        n_row = self.train_X.shape[0]
        distance = [self.cal_dist(x, self.train_X[i,:]) for i in xrange(n_row)]
        distance = np.array(distance)
        index = np.argsort(distance)
        sum_neighbor = sum(self.train_y[index[:self.k]])
        return 1. if sum_neighbor > 0 else -1.


    def predict(self, X):
        """
        find neighbor
        """
        n_row = X.shape[0]
        pred = [self._single_predict(X[i,:]) for i in xrange(n_row)]
        return np.array(pred)


def experiment(n_neighbor):
    train_X, train_y = load_file(TRAIN_FILE)
    test_X, test_y = load_file(TEST_FILE)
    # KNN
    clf = KNN(k=n_neighbor, distance_func=euclidean_distance)
    clf.fit(train_X, train_y)
    train_pred = clf.predict(train_X)
    train_acc = float(sum(train_pred == train_y)) / train_X.shape[0]

    test_pred = clf.predict(test_X)
    test_acc = float(sum(test_pred == test_y)) / test_X.shape[0]

    print("train accuracy: %.3f" % train_acc)
    print("est accuracy: %.3f" % test_acc)

def main():
     # 1-NN
    experiment(1)

    # 5-NN
    experiment(5)
