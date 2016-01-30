import numpy as np
from collections import Counter

TRAIN_FILE = ''
TEST_FILE = ''


def load_file(file_name):
    with open(file_name, 'rb') as f:
        lines = f.readlines()
        lines = [l.split(' ') for l in lines]
        lines = [filter(lambda x: len(x) > 0, l) for l in lines]
        lines = [filter(lambda x: len(x) > 0, l) for l in lines]
        lines = [[float(i) for i in line] for line in lines]
        y = np.array([l[-1] for l in lines])
        X = np.array([l[:-1] for l in lines])
    return X, y


class DTree(object):
    """
    decision tree with gini index
    """

    def __init__(self, level):
        self.level = level
        self.left = None # left tree
        self.right = None # right tree

        # for branch node
        self.branch_params = None
        # for leaf node
        self.value = None
        print('create Node in level %d' % level)
        
    def single_pred(self, record):
        """
        pred class os instance X
        """
        if self.value:
            return self.value
        else:
            # goes to sub tree
            feature_idx, theta = self.branch_params
            branch = record[feature_idx] - theta
            if branch < 0:
                return self.left.single_pred(record)
            else:
                return self.right.single_pred(record)

    def predict(self, X):
        n, _ = X.shape
        ret = [self.single_pred(X[i]) for i in xrange(n)]
        return np.array(ret)

    def node_count(self):
        """
        return sub node count in tree
        """
        if self.value:
            cnt = 0
        else:
            left_cnt = self.left.node_count()
            right_cnt = self.right.node_count()
            cnt = 1 + left_cnt + right_cnt
        return cnt

    @classmethod
    def _terminate(cls, y):
        if len(y) < 2 or np.unique(y).shape[0] == 1:
            ret = True
        else:
            ret = False
        if ret:
            print('terminate condtion met')
        return ret

    @classmethod
    def _gini_index(cls, ys):
        ys = np.array(ys)
        total = ys.sum()
        norm_y = ys / float(total)
        gini_idx = 1. - np.power(norm_y, 2).sum()
        return gini_idx

    @classmethod
    def _best_split(cls, X, y):
        """
        return tuple (feature_idx, sign, theta)
        """
        n = X.shape[0]
        num_feature = X.shape[1]
        y_types = np.unique(y)

        # initialize
        min_score = float(n)
        feature_idx = None
        best_theta = None
        best_idx = None

        for feature_idx in xrange(num_feature):
            # counter for y
            cumulate_y = Counter()
            rest_y = Counter()
            for y_type in y_types:
                cnt = np.where(y == y_type)[0].shape[0]
                rest_y[y_type] = cnt

            # sorted data
            sorted_idx = np.argsort(X[:, feature_idx])
            sorted_X = np.copy(X)
            sorted_y = np.copy(y)
            sorted_X = sorted_X[sorted_idx]
            sorted_y = sorted_y[sorted_idx]
            #print "_best_split:", sorted_X.shape, sorted_y.shape

            for idx in xrange(n-1):
                theta = (sorted_X[idx, feature_idx] + sorted_X[idx + 1, feature_idx]) / 2
                y_label = sorted_y[idx]
                cumulate_y[y_label] += 1
                rest_y[y_label] -= 1
                left_cnt = sum(cumulate_y.values())
                right_cnt = sum(rest_y.values())
                w_1 = left_cnt * cls._gini_index(cumulate_y.values())
                w_2 = right_cnt * cls._gini_index(rest_y.values())
                score = w_1 + w_2
                if score < min_score:
                    min_score = score
                    best_theta = theta
                    best_idx = feature_idx
                    #print('new min score: %.3f' % score)
                    #print('feature: %d, theta: %.3f' % (best_idx, best_theta))
                    #print('left: %d, right: %d' % (left_cnt, right_cnt))
        print('feature: %d, theta: %.3f' % (best_idx, best_theta))
        return (best_idx, best_theta)

    def _data_split(self, X, y):
        feature_idx, theta = self.branch_params
        X_feature = X[:, feature_idx]
        pos_idx = np.where(X_feature > theta)[0]
        neg_idx = np.where(X_feature < theta)[0]
        #print X_feature.shape
        #print pos_idx
        #print neg_idx
        X_pos = X[pos_idx]
        y_pos = y[pos_idx]
        X_neg = X[neg_idx]
        y_neg = y[neg_idx]

        msg = 'split data: pos=%d, %d, neg=%d, %d' % (X_pos.shape[0], y_pos.shape[0], X_neg.shape[0], y_neg.shape[0])
        print(msg)
        return X_pos, y_pos, X_neg, y_neg

    def fit(self, X, y):
        if self._terminate(y):
            self.value = y[0]
            # print "terminate: pred=%d" % self.value
        else:
            if self.level > 10:
                self.value = y[0]
                return self

            self.branch_params = self._best_split(X, y)
            self.left = DTree(self.level + 1)
            self.right = DTree(self.level + 1)

            pos_X, pos_y, neg_X, neg_y = self._data_split(np.copy(X), np.copy(y))
            # neg sample in left tree, pos in right tree
            print('fit left tree, level %d' %self.left.level)
            self.left.fit(neg_X, neg_y)
            print('fit right tree, level %d' %self.right.level)
            self.right.fit(pos_X, pos_y)
        return self


class DecisionStump(object):
    def __init__(self, level):
        self.branch_params = None
        # for leaf node
        self.value = None
        # print('create Node in level %d' % level)
        
    def single_pred(self, record):
        """
        pred class os instance X
        """
        # goes to sub tree
        sign, feature_idx, theta = self.branch_params
        branch = sign * (record[feature_idx] - theta)
        return -1. if branch < 0 else 1.

    def predict(self, X):
        n, _ = X.shape
        ret = [self.single_pred(X[i]) for i in xrange(n)]
        return np.array(ret)

    @classmethod
    def _gini_index(cls, ys):
        ys = np.array(ys)
        total = ys.sum()
        norm_y = ys / float(total)
        gini_idx = 1. - np.power(norm_y, 2).sum()
        return gini_idx

    @classmethod
    def _best_split(cls, X, y):
        """
        return tuple (feature_idx, sign, theta)
        """
        n = X.shape[0]
        num_feature = X.shape[1]
        y_types = np.unique(y)

        # initialize
        min_score = float(n)
        feature_idx = None
        best_theta = None
        best_idx = None

        for feature_idx in xrange(num_feature):
            # counter for y
            cumulate_y = Counter()
            rest_y = Counter()
            for y_type in y_types:
                cnt = np.where(y == y_type)[0].shape[0]
                rest_y[y_type] = cnt

            # sorted data
            sorted_idx = np.argsort(X[:, feature_idx])
            sorted_X = np.copy(X)
            sorted_y = np.copy(y)
            sorted_X = sorted_X[sorted_idx]
            sorted_y = sorted_y[sorted_idx]
            #print "_best_split:", sorted_X.shape, sorted_y.shape

            for idx in xrange(n-1):
                theta = (sorted_X[idx, feature_idx] + sorted_X[idx + 1, feature_idx]) / 2
                y_label = sorted_y[idx]
                cumulate_y[y_label] += 1
                rest_y[y_label] -= 1
                left_cnt = sum(cumulate_y.values())
                right_cnt = sum(rest_y.values())
                w_1 = left_cnt * cls._gini_index(cumulate_y.values())
                w_2 = right_cnt * cls._gini_index(rest_y.values())
                score = w_1 + w_2
                if score < min_score:
                    min_score = score
                    best_theta = theta
                    best_idx = feature_idx
                    #print('new min score: %.3f' % score)
                    #print('feature: %d, theta: %.3f' % (best_idx, best_theta))
                    #print('left: %d, right: %d' % (left_cnt, right_cnt))
        print('feature: %d, theta: %.3f' % (best_idx, best_theta))
        return (best_idx, best_theta)

    def _stump_sign(self, X, y, feature_idx, theta):
        X_feature = X[:, feature_idx]
        pos_idx = np.where(X_feature > theta)[0]
        neg_idx = np.where(X_feature < theta)[0]
        #print X_feature.shape
        #print pos_idx
        #print neg_idx
        return 1. if y[pos_idx].sum() > 0.0 else -1.


    def fit(self, X, y):
        feature_idx, theta = self._best_split(X, y)
        sign = self._stump_sign(X, y, feature_idx, theta)
        self.branch_params = (sign, feature_idx, theta)
        return self


def main():
    X_train, y_train = load_file(TRAIN_FILE)
    X_test, y_test = load_file(TEST_FILE)
    d_tree = DTree(0)
    d_tree.fit(X_train, y_train)

    train_pred = d_tree.predict(X_train)
    train_acc = (y_train == train_pred).sum() / float(len(y_train))
    test_pred = d_tree.predict(X_test)
    test_acc = (y_test == test_pred).sum() / float(len(y_test))

    print train_acc, test_acc

    return d_tree

