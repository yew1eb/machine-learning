import numpy as np

TRAIN_FILE = ''


def load_file(file_name):
    with open(file_name, 'rb') as f:
        lines = f.readlines()
        lines = [l.strip('\n').split(' ') for l in lines]
        lines = [filter(lambda x: len(x) > 0, l) for l in lines]
        lines = [[float(i) for i in line] for line in lines]
        X = np.array(lines)
    return X


def euclidean_distance(x_1, x_2):
    """
    euclidean_distance. can ignore `np.sqrt` in this case to return L2 cost directly.
    """
    if x_1.shape != x_2.shape:
        raise ValueError("shape mismatch")
    diff = x_1 - x_2
    distance = np.sqrt(sum(np.power(diff, 2)))
    return distance


class Kmeans(object):

    """
    Kmean with EM udpate.
    """

    def __init__(self, k, dist_func):
        self.k = k
        self.dist_func = dist_func

    def random_pick_center(self, X):
        n_row = X.shape[0]
        k = self.k
        idx = np.random.choice(xrange(n_row), k, replace=False)
        centers = X[idx, :].copy()
        return centers

    def group_idx(self, x):
        """
        return nearest indx group
        """
        centers = self.centers
        dist = [self.dist_func(x, center) for center in centers]
        dist = np.array(dist)
        group = np.argmin(dist)
        return group

    def _e_step(self, X):
        """
        fix center, assign group
        """
        n_row = X.shape[0]
        cluster_idx = [self.group_idx(X[i, :]) for i in xrange(n_row)]
        cluster_idx = np.array(cluster_idx)
        return cluster_idx

    def _m_step(self, X, clusters):
        """
        update centers
        """
        k = self.k
        new_centers = []
        for i in xrange(k):
            idx = np.where(clusters == i)[0]
            cluster_X = X[idx, :]
            cluster_size = cluster_X.shape[0]
            new_center = cluster_X.sum(axis=0) / cluster_size
            new_centers.append(new_center)

            #print("group %d: size: %d" % (i, cluster_size))
            #print("new_center: %s" % new_center)
        new_centers = np.array(new_centers)
        return new_centers

    def cal_cost(self, X, groups):
        """
        return cost of all clusters
        """
        k = self.k
        total_cost = 0.
        for i in xrange(k):
            idx = np.where(groups == i)
            group_X = X[idx, :]
            diff = group_X - self.centers[i, :]
            cost = np.power(diff, 2).sum()
            total_cost += cost
        avg_cost = total_cost / X.shape[0]
        return avg_cost

    def fit(self, X, iterations=15):
        self.centers = self.random_pick_center(X)

        for n_iter in xrange(iterations):
            cluster_idx = self._e_step(X)
            # calcualte cost
            cost = self.cal_cost(X, cluster_idx)
            print("iteration %d: cost: %.3f" % (n_iter, cost))

            new_centers = self._m_step(X, cluster_idx)
            self.centers = new_centers

        # result
        cluster_idx = self._e_step(X)
        cost = self.cal_cost(X, cluster_idx)
        print("iteration: %d, cost: %.3f" % (iterations, cost))

        return self

    def transform(self, X):
        """
        assign cluster number for X
        """
        return self._e_step(X)

    def fit_transform(self, X, iterations=15):
        """
        fit then transform
        """
        return self.fit(X, iterations=iterations).transform(X)


def main():
    train_X = load_file(TRAIN_FILE)
    n_test = 100
    cost_results = []
    for _ in xrange(n_test):
        transformer = Kmeans(k=10, dist_func=euclidean_distance)
        cluster_idx = transformer.fit_transform(train_X, iterations=10)
        cost = transformer.cal_cost(train_X, cluster_idx)
        cost_results.append(cost)

    return cost_results
