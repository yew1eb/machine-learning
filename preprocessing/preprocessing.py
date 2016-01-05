import numpy as np

def normalize(X, norm='l2', axis=1):
    """scales individual samples to have unit norm."""
    if axis == 0:
        X = X.T

    if norm == 'l1':
        norms = np.abs(X).sum(axis=1)
    else:
        norms = np.sqrt((X * X).sum(axis=1))
    new = X / norms[:, np.newaxis]

    if axis == 0:
        new = new.T

    return new

def scale(X, axis=0):
    new = X - np.mean(X, axis=0)
    return new / np.std(new, axis=0)

def main():
    # a = np.array([[1,2,3],[3,4,5]])
    # print(normalize(a, axis=0))
    X = np.array([[ 1., -1.,  2.],
                  [ 2.,  0.,  0.],
                  [ 0.,  1., -1.]])
    print(scale(X))

if __name__ == '__main__': main()