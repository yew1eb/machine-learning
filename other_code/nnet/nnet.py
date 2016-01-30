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


def d_tanh(x):
    """
    derivative of tanh(x) = 1. - (tanh(x) ^.2)
    """
    return 1. - np.power(np.tanh(x), 2)


class NNet(object):

    """
    multi-layer Neural Network with SGD training and tanh transformation.
    """

    def __init__(self, layers, r_min, r_max, learn_rate):
        """
        layers should be list of neurons in each layer, ex. [2, 3, 1]
        """

        if not isinstance(layers, list) or len(layers) < 3:
            raise ValueError('invalid layer parammeter')
        self.layers = layers
        n_layer = len(layers)
        self.n_layer = n_layer
        self.r_min = r_min
        self.r_max = r_max
        self.learn_rate = learn_rate

        # initialize ws
        self.ws = []
        for layer_idx in range(n_layer - 1):
            layer_size = (layers[layer_idx] + 1, layers[layer_idx + 1])
            w = self.init_w(layer_size)
            self.ws.append(w)

    def init_w(self, size):
        """
        initialize wight from uniform distribution
        """
        return np.random.uniform(self.r_min, self.r_max, size=size)

    def forward_prop(self, x):
        """
        forward propagation
        """
        scores = []
        xs = []
        xs.append(x)

        # input layer
        for layer_idx in range(self.n_layer - 1):
            x_add_1 = np.append(1., xs[-1])
            w = self.ws[layer_idx]
            # remove later
            # print x_add_1.shape, w.shape
            assert(x_add_1.shape[0] == w.shape[0])
            next_score = np.dot(x_add_1, w)
            next_x = np.tanh(next_score)
            scores.append(next_score)
            xs.append(next_x)
        return xs, scores

    def backward_prop(self, xs, scores, y):
        """
        backward propagation for 1 record.
        """
        deltas = []
        # output layer
        # print xs[-1].shape
        #assert(xs[-1].shape == ())
        #assert(scores[-1].shape == ())

        deltas = []
        delta_L = -2 * (y - xs[-1]) * d_tanh(xs[-1])
        # use reverse order first. reverse in the end
        deltas.append(delta_L)

        for layer_idx in range(self.n_layer - 1, 0, -1):
            prev_layer_idx = layer_idx - 1
            w = self.ws[prev_layer_idx]
            x = xs[prev_layer_idx]
            no_bias_w = w[1:, ]
            delta = deltas[-1]
            #print
            #print no_bias_w.shape
            #print delta.shape
            #print x.shape
            #print 
            prev_delta = np.dot(no_bias_w, delta) * d_tanh(x)
            deltas.append(prev_delta)

        deltas.reverse()
        return deltas

    def sgd(self, X, y):
        xs, scores = self.forward_prop(X)
        deltas = self.backward_prop(xs, scores, y)

        new_ws = []
        for layer_idx in xrange(self.n_layer - 1):
            old_w = self.ws[layer_idx]
            x = xs[layer_idx]
            x_add_1 = np.append(1., x)
            delta = deltas[layer_idx + 1]
            diff = np.outer(x_add_1, delta)
            assert(old_w.shape == diff.shape)
            new_w = old_w - self.learn_rate * diff
            new_ws.append(new_w)
        self.ws = new_ws

    def fit(self, X, y, iterations):
        n_row = X.shape[0]
        for iter_idx in xrange(iterations):
            update_idx = np.random.randint(0, n_row)
            self.sgd(X[update_idx, :], y[update_idx])

            #if iter_idx > 0 and iter_idx % 1000 == 0:
            #    # calculate predict score
            #    iter_error = self.cal_error(X, y)
            #    msg = "iteration: %d, error: %.3f" % (iter_idx, iter_error)
            #   print(msg)
        #error = self.cal_error(X, y)
        #msg = "end iteration, error: %.3f" % error
        #print(msg)


    def predict(self, X):
        n_row = X.shape[0]
        current_x = X
        for w in self.ws:
            bias = np.ones((n_row, 1))
            x_add_1 = np.hstack([bias, current_x])
            score = np.dot(x_add_1, w)
            current_x = np.tanh(score)
        return current_x[:, 0]

    def cal_r2_error(self, X, y):
        pred_X = self.predict(X)
        diff = y - pred_X
        error = np.power(diff, 2).sum()
        error /= len(y)
        return error

    def cal_error(self, X, y):
        pred_X = self.predict(X)
        pred_y = np.ones((X.shape[0],))
        pred_y[np.where(pred_X < 0.0)] = -1.
        err = 1. - float((y == pred_y).sum()) / len(y)
        return err

def main():
    train_X, train_y = load_file(TRAIN_FILE)
    test_X, test_y = load_file(TEST_FILE)

    # Q11: M = 1,6,11,16,21
    # Q12: r = {0,0.001,0.1,10,1000}
    # Q13: learn_rate = {0.001,0.01,0.1,1,10}
    params = {
        'layers': [2, 8, 3, 1],
        'r_min': -.1,
        'r_max': .1,
        'learn_rate': .01,
    }

    n_experiment = 50
    train_errs = []
    test_errs = []
    for i in xrange(n_experiment):
        clf = NNet(**params)
        clf.fit(train_X, train_y, iterations=50000)

        train_err = clf.cal_error(train_X, train_y)
        test_err = clf.cal_error(test_X, test_y)
        train_errs.append(train_err)
        test_errs.append(test_err)
        print("experiment %d, test err: %.3f" % (i, test_err))

    #print(params)
    print("mean train err: %.6f" % np.mean(train_errs))
    print("mean test err: %.6f" % np.mean(test_errs))
