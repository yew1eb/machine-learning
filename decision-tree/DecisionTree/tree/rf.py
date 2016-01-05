import numpy as np

TRAIN_FILE = ''
TEST_FILE = ''

from dtree import DTree

def boot_strap(X, y):
    n = X.shape[0]
    sample_idx = np.random.choice(n, n, replace=True)

    new_X = X[sample_idx]
    new_y = y[sample_idx]
    return new_X, new_y


def rf_prediction(trees, X):
    pred_sum = np.zeros((X.shape[0],))
    for tree in trees:
        pred = tree.predict(X)
        pred_sum += pred

    pos_idx = np.where(pred_sum >= 0.)[0]
    neg_idx = np.where(pred_sum < 0.)[0]

    pred_sum[pos_idx] = 1.
    pred_sum[neg_idx] = -1.
    return pred_sum


def RF(X_train, y_train):
    """
    random forest
    """

    trees = []
    for i in xrange(300):
        d_tree = DTree(0)
        bootstrap_X, bootstrap_y = boot_strap(X_train, y_train)
        d_tree.fit(bootstrap_X, bootstrap_y)
        trees.append(d_tree)
    return trees

def stump(X_train, y_train):
    trees = []
    for i in xrange(300):
        d_tree = DecisionStump(0)
        bootstrap_X, bootstrap_y = boot_strap(X_train, y_train)
        d_tree.fit(bootstrap_X, bootstrap_y)
        trees.append(d_tree)
    return trees

def experiment():
    X_train, y_train = load_file(TRAIN_FILE)
    X_test, y_test = load_file(TEST_FILE)

    train_ret = []
    test_ret = []
    for i in xrange(10):
        #trees = RF(X_train, y_train)
        trees = stump(X_train, y_train)

        train_pred = rf_prediction(trees, X_train)
        train_acc = (y_train == train_pred).sum() / float(len(y_train))

        test_pred = rf_prediction(trees, X_test)
        test_acc = (y_test == test_pred).sum() / float(len(y_test))

        train_ret.append(train_acc)
        test_ret.append(test_acc)
        print train_acc, test_acc
